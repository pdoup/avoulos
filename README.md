# Analysis of the Greek Parliament Proceedings '89 – '20
Big Data Analytics Project - Fall 2021  - [Link to *iMEdD* GitHub repo](https://github.com/iMEdD-Lab/Greek_Parliament_Proceedings)

![pap](https://www.in.gr/wp-content/uploads/2019/11/Vouli_EMEAgr_980x620_02.jpg)
---

* [X] Task 1 : Given all speeches (for all years) we need to detect the different topics (i.e., thematic areas), most important keywords and how they change across years
* [X] Task 2 : Given all speeches we need to detect pairwise similarities between parliament members & detect the top-k pairs with the highest degree of similarity
* [X] Task 3 : For each member and also for each party we need to detect how the most important keywords evolve across years.
* [X] Task 4 : Detect any significant deviation (per member, per party or in general) with respect to the speeches before and after the crisis
* [X] Task 5 : Taking into account all speeches, we need to detect if we can group them in meaningful clusters.Check about the participation of each member in each cluster and    also the participation of each party in the cluster.
* [X] Task 6 : % of Male/Female positions in the parliament over the years 

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

---

### **Some useful links**
- [Spark ML Library Documetation](https://spark.apache.org/docs/3.0.1/ml-guide.html)
- [Spark Tutorial](https://www.tutorialspoint.com/apache_spark/index.htm)
- [Spark Tutorial YT](https://www.youtube.com/watch?v=S2MUhGA3lEw)
