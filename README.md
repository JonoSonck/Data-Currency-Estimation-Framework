# Data Currency Estimation Framework

This repository contains the code linked to a research project about data currency estimation. 

A paper regarding the theory behind the code has been submitted to the IPMU26 conference.

**Paper Abstract:**
Data currency is a dimension in data quality research that addresses discrepancies related to a temporal decline between stored data and real-world values. While several methods exist to estimate data currency, existing research on this topic remains fragmented and approaches the problem from several perspectives. In this paper, we propose a unified framework that integrates existing estimation techniques by representing them as components of nodes within a Bayesian belief network. Each node maintains attribute-specific (meta-)data and, where applicable, an estimation technique for modeling the probability distribution over an attribute’s age. The edges impose dependencies and constraints on the (transfer of) belief through the network. We further describe the combination of nodes in the overall network architecture, including mechanisms for handling dependencies and aggregation. We demonstrate the possibilities of the proposed framework by replicating existing techniques and further expanding them. The modularity of our network allows us to adapt the inner workings of our nodes to accommodate the characteristics of the data under consideration, all while maintaining ease-of-use and interpretability.


*The link to the full paper will be shared when it's accepted and published.*


## Repository file structure

- cur_est_net_testing_example: a basic usage example which shows the necessary steps to perform the currency estimates using the framework.
 
:file_folder:  src
   - :file_folder:  changepoint: contains all estimation methods regarding *'change point detection'* methods, including (1) CUSUM change point detection.
   - :file_folder:  dependency: contains all belief propagation methods related to connected nodes, including (1) aggregator which combines multiple nodes into a single belief.
   - :file_folder:  shelflife: contains all estimation methods regarding *'shelf-life'* models, including (1) basic shelf-life with hazard function (2) dynamic shelf-life and (3) conditional shelf-life that bases its hazard value on other data values.
   - abstract_node: implements the basic structure of nodes, including a specific *age node* and *data node* implementation. The files above use this structure for further development. 
   - currency_network: the implementation of the network architecture, which connects the nodes into a Bayesian belief network structure, and instructs each node to perform the right measures in sequential order.


## Usage explanation

The *'cur_est_net_testing_example'* notebook gives a code-based overview of the necessary steps to perform the estimates using our proposed framework. In summary, the user has to:
1. Load in the data as a pandas DataFrame.
2. Instantiate the preferred node class for each attribute that is involved in the estimate.
3. Instantiate the network with all relevant nodes.
4. Run the network estimation function (clear after each estimate!) and output the results.



## Contributers

| Name                        | ORCID    | email             |
| ----------------            | :------: |  ---------------- |
| Jono Sonck                  | 0009-0001-0193-7853  | jono.sonck@ugent.be |
| Antoon Bronselaer           | 0000-0001-6663-192X  | antoon.bronselaer@ugent.be |
| Guy De Tré                  | 0000-0002-1283-1915  | guy.detre@ugent.be |

All researchers are part of the [Database, Document and Content Management research group](https://ddcm.ugent.be/)  of the [Telecommunications and Information Processing department](https://telin.ugent.be/telin/) in the [Faculty of Engineering and Architecture](https://www.ugent.be/ea/en) at [Ghent University](https://www.ugent.be/).
