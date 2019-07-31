#include "parallelgraph.h"

#include "compression.h"
#include "BayesRBase.hpp"
#include "kernel.h"
#include "markerbuilder.h"

#include <iostream>

ParallelGraph::ParallelGraph(size_t maxDecompressionTokens, size_t maxAnalysisTokens)
    : AnalysisGraph()
    , m_graph(new graph)
    , m_decompressionTokens(maxDecompressionTokens)
    , m_analysisTokens(maxAnalysisTokens)
{
    m_decompressionJoinNode.reset(new decompression_join_node(*m_graph));

    // Decompress the column for this marker then process the column using the algorithm class
    auto f = [this] (DecompressionTuple tuple) -> DecompressionTuple {
        auto &msg = std::get<1>(tuple);
        // Decompress the column
        std::unique_ptr<MarkerBuilder> builder{m_bayes->markerBuilder()};
        builder->initialise(msg.snp, msg.numInds);
        const auto index = m_bayes->indexEntry(msg.snp);
        if (m_bayes->compressed()) {
            builder->decompress(m_bayes->compressedData(), index);
        } else {
            builder->read(m_bayes->preprocessedFile(), index);
        }
        msg.kernel = m_bayes->kernelForMarker(builder->build());
        return tuple;
    };

    m_decompressionNode.reset(new decompression_node(*m_graph,
                                                     m_decompressionNodeConcurrency,
                                                     f));

    m_analysisJoinNode.reset(new analysis_join_node(*m_graph));

    // Sampling of the column to the async algorithm class
    auto g = [this] (AnalysisTuple tuple) -> AnalysisTuple {
        auto &msg = std::get<1>(std::get<1>(tuple));
        msg.result = m_bayes->processColumnAsync(msg.kernel.get());
        return tuple;
    };

    m_analysisNode.reset(new analysis_node(*m_graph, m_analysisNodeConcurrency, g));

    // Decide whether to continue calculations or discard
    auto h = [] (decision_node::input_type input,
            decision_node::output_ports_type &outputPorts) {

        auto &decompressionTuple = std::get<1>(input);
        auto &msg = std::get<1>(decompressionTuple);

        if (msg.result->betaOld != 0.0 || msg.result->beta != 0.0) {
            // Do global computation
            std::get<2>(outputPorts).try_put(input);
        } else {
            // Discard
            std::get<0>(outputPorts).try_put(std::get<0>(decompressionTuple));
            std::get<1>(outputPorts).try_put(std::get<0>(input));
        }
    };

    m_decisionNode.reset(new decision_node(*m_graph, unlimited, h));

    // Do global computation
    auto i = [this] (global_update_node::input_type input,
            global_update_node::output_ports_type &outputPorts) {

        auto &decompressionTuple = std::get<1>(input);
        auto &msg = std::get<1>(decompressionTuple);

        m_bayes->updateGlobal(msg.kernel.get(),
                              msg.result->betaOld,
                              msg.result->beta,
                              *msg.result->deltaEpsilon);

        std::get<0>(outputPorts).try_put(std::get<0>(decompressionTuple));
        std::get<1>(outputPorts).try_put(std::get<0>(input));
    };
    // Use the serial policy
    m_globalUpdateNode.reset(new global_update_node(*m_graph, serial, i));

    // Force synchronisation after m_anaylsisToken analyses
    auto j = [this](AnalysisToken t) -> continue_msg {
        (void) t; // Unused
        --m_analysisTokenCount;

        if (m_analysisTokenCount == 0) {
            // Allow the next set of analyses to take place
            queueAnalysisTokens();
        }

        return continue_msg();
    };
    m_analysisControlNode.reset(new analysis_control_node(*m_graph, serial, j));


    // Set up the graph topology:
#if defined(TBB_PREVIEW_FLOW_GRAPH_TRACE)
    m_graph->set_name("ParallelGraph");
    m_analysisNode->set_name("analysis_node");
    m_decisionNode->set_name("decision_node");
    m_globalUpdateNode->set_name("global_update_node");
    m_decompressionJoinNode->set_name("decompression_join_node");
    m_decompressionNode->set_name("decompression_node");
    m_analysisJoinNode->set_name("analysis_join_node");
    m_analysisControlNode->set_name("analysis_control_node");
#endif

    make_edge(*m_decompressionJoinNode, *m_decompressionNode);
    make_edge(*m_decompressionNode, input_port<1>(*m_analysisJoinNode));
    make_edge(*m_analysisJoinNode, *m_analysisNode);
    make_edge(*m_analysisNode, *m_decisionNode);
    make_edge(output_port<0>(*m_decisionNode), input_port<0>(*m_decompressionJoinNode));
    make_edge(output_port<1>(*m_decisionNode), *m_analysisControlNode);
    make_edge(output_port<2>(*m_decisionNode), *m_globalUpdateNode);
    make_edge(output_port<0>(*m_globalUpdateNode), input_port<0>(*m_decompressionJoinNode));
    make_edge(output_port<1>(*m_globalUpdateNode), *m_analysisControlNode);
}

void ParallelGraph::exec(BayesRBase *bayes,
                              unsigned int numInds,
                              unsigned int numSnps,
                              const std::vector<unsigned int> &markerIndices)
{
    if (!bayes) {
        std::cerr << "Cannot run ParallelGraph without bayes" << std::endl;
        return;
    }

    // Set our Bayes for this run
    m_bayes = bayes;

    // Do not allow Eigen to parallalize during ParallelGraph execution.
    const auto eigenThreadCount = Eigen::nbThreads();
    Eigen::setNbThreads(0);

    // Reset the graph from the previous iteration.
    m_graph->reset();
    queueDecompressionTokens();
    queueAnalysisTokens();

    // Push some messages into the top of the graph to be processed - representing the column indices
    for (unsigned int i = 0; i < numSnps; ++i) {
        Message msg = { i, markerIndices[i], numInds };
        input_port<1>(*m_decompressionJoinNode).try_put(msg);
    }

    // Wait for the graph to complete
    m_graph->wait_for_all();

    // Turn Eigen threading back on.
    Eigen::setNbThreads(eigenThreadCount);

    // Clean up
    m_bayes = nullptr;
}

size_t ParallelGraph::decompressionNodeConcurrency() const
{
    return m_decompressionNodeConcurrency;
}

void ParallelGraph::setDecompressionNodeConcurrency(size_t c)
{
    m_decompressionNodeConcurrency = c;
}

size_t ParallelGraph::decompressionTokens() const
{
    return m_decompressionTokens;
}

void ParallelGraph::setDecompressionTokens(size_t t)
{
    m_decompressionTokens = t;
}

size_t ParallelGraph::analysisNodeConcurrency() const
{
    return m_analysisNodeConcurrency;
}

void ParallelGraph::setAnalysisNodeConcurrency(size_t c)
{
    m_analysisNodeConcurrency = c;
}

size_t ParallelGraph::analysisTokens() const
{
    return m_analysisTokens;
}

void ParallelGraph::setAnalysisTokens(size_t t)
{
    m_analysisTokens = t;
}

void ParallelGraph::queueDecompressionTokens()
{
    for(DecompressionToken t = 0; t < m_decompressionTokens; ++t)
        input_port<0>(*m_decompressionJoinNode).try_put(t);
}

void ParallelGraph::queueAnalysisTokens()
{
    for(AnalysisToken t = 0; t < m_analysisTokens; ++t)
        input_port<0>(*m_analysisJoinNode).try_put(t);

    m_analysisTokenCount = m_analysisTokens;
}
