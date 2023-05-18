# SalesmanManager
Cluster and Rate the Customers for Factory Salesmen in According to Different Factors Using Reinforcement Learning.
Assume you have a factory, and you hire 10 salesman to visit the stores in a big city and introduce or sale your products. For making the optimum path for each salesman, we should do several works:

First, we should cluster the region, so each cluster would be dedicated to each salesman. This job is done by inspiring from Voronoi diagram.

Second, we should rank the stores of the reion. For example, if there is a store who didn't pay their debt, should not be visited sooner than a well reputated store. If a store have been bought a lot of goods recently, the salesman should put the store in above of their list. So it is obvious that we should rank the stores, each morning that the salesman wants to start their work. After a round is completed, the salesman can score the ranking. The program uses Reinforcement Learning to improve the ranking.
