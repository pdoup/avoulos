# Analysis of the Greek Parliament Proceedings '89 – '20
Big Data Analytics Project - Fall 2021  - [Link to *iMEdD* GitHub repo](https://github.com/iMEdD-Lab/Greek_Parliament_Proceedings)

![pap](https://thesocialist.gr/wp-content/uploads/2021/06/papandreou3_2306.jpg)
---

* [ ] Task 1 : Given all speeches (for all years) we need to detect the different topics (i.e., thematic areas), most important keywords and how they change across years
* [ ] Task 2 : Given all speeches we need to detect pairwise similarities between parliament members & detect the top-k pairs with the highest degree of similarity
* [ ] Task 3 : For each member and also for each party we need to detect how the most important keywords evolve across years.
* [ ] Task 4 : Detect any significant deviation (per member, per party or in general) with respect to the speeches before and after the crisis
* [ ] Task 5 : Taking into account all speeches, we need to detect if we can group them in meaningful clusters.Check about the participation of each member in each cluster and    also the participation of each party in the cluster.
* [ ] Task 6 : TBD

---

> **How to package and run a spark application**

1. Run ``` package ``` from the ``` sbt shell ``` (IntelliJ)
2. Once the ``` .jar ``` is created in the ``` target ``` folder run this command once inside that folder
```bash
spark-submit \
  --class <Name of the main class> \
  --master local[*] \  
  --executor-memory 8G \
  --total-executor-cores 4 \
  /path/to/examples.jar <add optional arguments here>
```
