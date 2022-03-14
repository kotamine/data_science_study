# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science with Databricks (basics)
# MAGIC 
# MAGIC Contents:
# MAGIC 1. File system commands
# MAGIC 2. Delta lake intro
# MAGIC 3. R data science (basics)
# MAGIC 4. Python data science (basics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1. File system commands

# COMMAND ----------

# MAGIC %fs ls /  

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs  # this is called from bash  # not available in community edition

# COMMAND ----------

ls /dbfs  # the same thing   -- not available in community edition

# COMMAND ----------

display(dbutils.fs.ls("/."))   # the same thing but this is called from python  

# COMMAND ----------

list_1 = [i for i in dbutils.fs.ls("/.")]  # for example, the paths can be put int a python list and extracted from it 
print('length =', len(list_1),'\n') 
print('first item file path =', list_1[0].path)

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets'))  # one can look at various datasets to play around with 

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets/flights'))

# COMMAND ----------

# for help with dbutils see: https://docs.databricks.com/_static/notebooks/dbutils.html
dbutils.fs.head('/databricks-datasets/flights/departuredelays.csv', 400) # maxBytes = 400

# COMMAND ----------

import pandas as pd
txt1 = dbutils.fs.head('/databricks-datasets/flights/departuredelays.csv', 400)
txt1_trunc = txt1[0:txt1.rfind('\n')]  # truncate this csv string at the last occurance of '\n' 

dat = [x.split(',') for x in txt1_trunc.split('\n')]  # convert it into a list
head_flight = pd.DataFrame(dat[1:], columns=dat[0])  # convert it into a pd.DataFrame 
head_flight   # pd.DataFrame is not the same thing as Apache Spark DataFrame

# COMMAND ----------

df_head_flight = spark.createDataFrame(head_flight) # convert it into a spark DataFrame 
display(df_head_flight.limit(5))

# COMMAND ----------

spark.sql("drop table if exists head_flight")
df_head_flight.write.saveAsTable('head_flight')
display(spark.sql("select * from head_flight "))

# COMMAND ----------

display(dbutils.fs.ls('/user/hive/warehouse'))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from head_flight 

# COMMAND ----------

from pyspark.sql.types import DoubleType
# add a numeric/double type delay column 
df_head_flight = df_head_flight.withColumn('delay_n', 
                                           df_head_flight['delay'].cast(DoubleType()))

# calculate mean delay by date and origin
display(df_head_flight['date','delay_n','origin'].groupBy('date','origin').mean('delay_n'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2. Delta Lake intro.
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/delta/intro-notebooks.html" target="_blank">databricks delta intro.</a>
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/data/tables.html#managed-and-unmanaged-tables" target="_blank">databricks managed and unmanaged tables</a> 

# COMMAND ----------

# DBTITLE 1,Read Databricks dataset  
# Define the input format and path.
read_format = 'delta'
load_path = '/databricks-datasets/learning-spark-v2/people/people-10m.delta'

# Load the data from its source.
people = spark.read \
    .format(read_format) \
    .load(load_path)

# Show the results.
display(people)

# COMMAND ----------

 people.__class__, people.count() # class = pyspark.sql.dataframe.DataFrame, 10M rows

# COMMAND ----------

[i for i in dir(people) if i[0]!='_'] # many panda-dataframe-like methods exist for spark dataframe

# COMMAND ----------

display(dbutils.fs.ls(load_path)) # 1 delta_log file and 16 parquet files

# COMMAND ----------

# DBTITLE 1,Write out DataFrame as Databricks Delta data
# Define the output format, output mode, columns to partition by, and the output path.
write_format = 'delta'
write_mode = 'overwrite'
partition_by = 'gender'
save_path = '/tmp/delta/people-10m'

# Write the data to its target.
people.write \
  .format(write_format) \
  .partitionBy(partition_by) \
  .mode(write_mode) \
  .save(save_path)

dbutils.fs.ls(save_path)

# COMMAND ----------

# DBTITLE 1,Unmanaged table
table_unmanaged = 'default.people10m_unmanaged'
query = "CREATE TABLE " + table_unmanaged + " USING DELTA LOCATION '" + save_path + "'"
print(query)

# Delete the table if exists.
spark.sql("DROP TABLE IF EXISTS " + table_unmanaged)

spark.sql(query) # create table
display(spark.sql("SELECT * from " + table_unmanaged))  # select from the table

# COMMAND ----------

display(spark.sql('DESCRIBE DETAIL people10m_unmanaged')) # see location is the specified source = dbfs:/tmp/delta/people-10m

# COMMAND ----------

spark.sql("drop table if exists " + table_unmanaged)  # dropping the unmanaged table
dbutils.fs.ls(save_path)  # doesn't delete the data

# COMMAND ----------

# DBTITLE 1,Temporary View
people.createOrReplaceTempView('people10m_vw')
display(spark.sql('select * from people10m_vw limit 1000'))

# COMMAND ----------

# DBTITLE 1,Save as a Managed table
sourceType = 'delta'
table_managed = 'people10m_managed'

# Delete the table if exists.
spark.sql("DROP TABLE IF EXISTS " + table_managed)

people.write \
  .format(sourceType) \
  .saveAsTable(table_managed)

display(spark.sql("SELECT * FROM " + table_managed))

# COMMAND ----------

display(spark.sql('DESCRIBE DETAIL people10m_managed')) # see location = dbfs:/user/hive/warehouse/people10m_managed

# COMMAND ----------

managed_location = 'dbfs:/user/hive/warehouse/'
display(dbutils.fs.ls(managed_location))

# COMMAND ----------

display(dbutils.fs.ls(managed_location + "/" + table_managed)) # 1 delta_log file + 2 parquet files

# COMMAND ----------

# Delete the saved data.
dbutils.fs.rm(save_path, True)

# COMMAND ----------

# display(spark.sql("SELECT * from " + table_unmanaged))  # it doesn't exist since the pointed source is deleted.

# COMMAND ----------

display(spark.sql("SELECT * FROM " + table_managed))  # it's still there in the managed location: dbfs:/user/hive/warehouse/people10m_managed

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS " + table_managed)
dbutils.fs.ls(managed_location)  # deletes the data in managed location 

# COMMAND ----------

# DBTITLE 1,Some visualization  
# visualize gender counts 
people.select('gender').orderBy('gender', ascending = False).groupBy('gender').count().display()

# COMMAND ----------

people.select("salary").orderBy("salary", ascending = False).display() 
# for this visualization, manually choose histogram, optionally chose bin size and resize/expand the chart 

# COMMAND ----------

# DBTITLE 1,Optimize table
display(spark.sql("OPTIMIZE " + table_name))

# COMMAND ----------

# DBTITLE 1,Show table history
display(spark.sql("DESCRIBE HISTORY " + table_name))

# COMMAND ----------

# Delete the managed table.
spark.sql("DROP TABLE " + table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. R data analysis and visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Interacting with Spark

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.1 DBI query
# MAGIC 
# MAGIC DBI allows for querying data on Spark SQL and putting it into the R session.  
# MAGIC 
# MAGIC Use case: you have data in Spark SQL and want to apply R's ggplot for data visualization  
# MAGIC - 1 Reduce data into key summary statistics via a SQL query 
# MAGIC - 2 Visualize the stats via ggplot

# COMMAND ----------

dbutils.fs.ls("/user/hive/warehouse")

# COMMAND ----------

spark.sql("create table people_v2 using delta location 'dbfs:/user/hive/warehouse/people10m'")

# COMMAND ----------

# MAGIC %r 
# MAGIC library(DBI) 
# MAGIC 
# MAGIC sc <- sparklyr::spark_connect(method = "databricks")
# MAGIC 
# MAGIC query = "select * from people_v2 limit 1000"  # pass a SQL query 
# MAGIC 
# MAGIC # evaluate the query and move all the results to R
# MAGIC result <- dbGetQuery(sc, query)
# MAGIC print(class(result))  # dbGetQuery()  returns a R data.frame object 
# MAGIC display(result)

# COMMAND ----------

# MAGIC %r
# MAGIC # another example
# MAGIC query1 <- "CREATE TABLE head_flights2 USING DELTA LOCATION 'dbfs:/user/hive/warehouse/head_flight/'"
# MAGIC 
# MAGIC dbGetQuery(sc, query1)
# MAGIC display(dbGetQuery(sc, 'select * from head_flights2'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3.1.2 SparkR 
# MAGIC 
# MAGIC SparkR creates and manipulates SparkDataFrame objects.
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/spark/latest/sparkr/overview.html" target="_blank">databricks SparkR intro.</a> 

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(magrittr)
# MAGIC 
# MAGIC cars93r <- read.df("/databricks-datasets/Rdatasets/data-001/csv/MASS/Cars93.csv",
# MAGIC                 source = "csv", header="true", inferSchema = "true")
# MAGIC 
# MAGIC display(head(cars93r))

# COMMAND ----------

# MAGIC %r
# MAGIC print(class(cars93r)) #  class = SparkDataFrame
# MAGIC methods(class=attributes(cars93r)$class[1])   # look at methods, inclusing agg, crosstab, filter, join, group_by, limit, etc.

# COMMAND ----------

# MAGIC %r 
# MAGIC cars93r %>% SparkR::crosstab('Manufacturer','Type')  # example of crosstab 

# COMMAND ----------

# MAGIC %r
# MAGIC display(
# MAGIC   cars93r %>% groupBy("Manufacturer") %>% SparkR::count() # example: groupBy + count
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC # example:  groupBy + aggregation + arrange/sort 
# MAGIC cars93r %>% 
# MAGIC   groupBy("Manufacturer") %>% 
# MAGIC   agg("price" = "avg") %>% 
# MAGIC   SparkR::arrange("Manufacturer") %>%
# MAGIC   display()

# COMMAND ----------

# MAGIC %r
# MAGIC saveAsTable(cars93r, 'cars93r', mode='overwrite') # save SparkDataFrame as a managed table

# COMMAND ----------

display(dbutils.fs.ls('user/hive/warehouse'))

# COMMAND ----------

# MAGIC %r
# MAGIC createOrReplaceTempView(cars93r, "cars93r_vw")  # create a view

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cars93r_vw

# COMMAND ----------

# MAGIC %r
# MAGIC # another example: convert R dataframe to SparkDataFrame and create a view for it
# MAGIC sp_faithful <- createDataFrame(faithful) # Converts R data.frame or list into SparkDataFrame.
# MAGIC createOrReplaceTempView(sp_faithful, "faithful_vw")  # create a view
# MAGIC dplyr::src_tbls(sc) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.3 sparklyr 
# MAGIC 
# MAGIC **SparkR** and **sparklyr** offer similar functionalities of data wrangling and machine learning function. 
# MAGIC 
# MAGIC The main difference is that **sparklyr** comes with **dplyr-functions** with the same syntax as those who are used to manipulating R dataframe with dplyr library.    
# MAGIC 
# MAGIC - <a href="https://docs.databricks.com/_static/notebooks/sparklyr.html" target="_blank">databricks sparklyr intro.</a>
# MAGIC - <a href="https://spark.rstudio.com/guides/dplyr.html" target="_blank">dplyr guides</a>

# COMMAND ----------

# MAGIC %r 
# MAGIC library(sparklyr)  

# COMMAND ----------

# MAGIC %r
# MAGIC print(paste("class of mtcars is:", class(mtcars)))  # note: mtcars is auto-loaded in R
# MAGIC head(mtcars)

# COMMAND ----------

# MAGIC %r 
# MAGIC mtcars$car_model = rownames(mtcars)
# MAGIC 
# MAGIC sc <- spark_connect(method = "databricks")
# MAGIC tbl_mtcars <- copy_to(sc, mtcars, overwrite=TRUE) # copy it in Spark 
# MAGIC print(paste("class of tbl_cars is:", class(tbl_mtcars)[1]))  # main class is tbl_spark 
# MAGIC cat("\n")
# MAGIC print("Spark Connection contains: ")
# MAGIC dplyr::src_tbls(sc)  # show tables in Spark Connection 

# COMMAND ----------

# MAGIC %r
# MAGIC tbl_mtcars %>% spark_write_parquet(path='user/hive/warehouse/mtcars', mode='overwrite')

# COMMAND ----------

# MAGIC %r
# MAGIC ?copy_to.spark_connection # help of copy_to

# COMMAND ----------

# MAGIC %r
# MAGIC methods(class=class(sc)[1]) # Spark connection has lots of machine learning methods

# COMMAND ----------

# MAGIC %r
# MAGIC methods(class=class(tbl_mtcars))
# MAGIC # methods for tbl_spark is those of spark connection + dplyr-like methods (filter, mutate, summarise, group_by, select, left_join) 

# COMMAND ----------

# MAGIC %r
# MAGIC # add a character-version of the cylinder variable in R data frame
# MAGIC mtcars <- mtcars %>% mutate(ch_cyl = as.character(cyl)) 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- query the mtcars copied to spark connection. "ch_cyl" does not exist here
# MAGIC select * from mtcars 

# COMMAND ----------

# MAGIC %r 
# MAGIC # add the same character-version variable in tbl_spark
# MAGIC # note: loading SparkR and dplyr libraries together may cause some namespace conflicts. "dplyr::"-prefix is used to explicitly call a function from dplyr. 
# MAGIC 
# MAGIC tbl_mtcars <- tbl_mtcars %>% dplyr::mutate(ch_cyl = as.character(cyl))
# MAGIC display(tbl_mtcars %>% dplyr::collect())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The new variable still does not exist. This may be counter-intuitive. 
# MAGIC select * from mtcars 

# COMMAND ----------

# MAGIC %r
# MAGIC # When you modify a tbl_spark object and want Spark SQL to reflect that change, re-register it in Spark.  
# MAGIC sdf_register(tbl_mtcars,'tbl_mtcars')
# MAGIC dplyr::src_tbls(sc)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tbl_mtcars 

# COMMAND ----------

# MAGIC %r
# MAGIC summary1 <- tbl_mtcars %>% 
# MAGIC   dplyr::group_by(ch_cyl) %>%
# MAGIC   dplyr::summarise(count = n(),
# MAGIC             avg_hp = mean(hp) %>% round(0),
# MAGIC             avg_mpg = mean(mpg) %>% round(1)) %>% 
# MAGIC   dplyr::arrange(desc(ch_cyl)) %>% 
# MAGIC   dplyr::collect()  # collect() converts into a R dataframe
# MAGIC 
# MAGIC display(summary1)

# COMMAND ----------

# MAGIC %r
# MAGIC library(ggplot2)
# MAGIC 
# MAGIC # utility function to adjust plot-display in databrics notebook 
# MAGIC plotprint <- function(plot_obj, width=7, height=4.5, just="top", x=4.0, y=8.0) { 
# MAGIC   print(plot_obj, vp=grid::viewport(width=width, height=height, just=just, x=x, y=y, default.units='inch'))
# MAGIC   }
# MAGIC 
# MAGIC # scatterplot of hp and mpg
# MAGIC pl <- tbl_mtcars %>% 
# MAGIC   ggplot(mapping = aes(x = hp, y = mpg, color=ch_cyl)) +
# MAGIC   geom_point(aes(shape = ch_cyl)) 
# MAGIC 
# MAGIC plotprint(pl)

# COMMAND ----------

# MAGIC %md
# MAGIC One could directly plot **tbl_spark  objects** in `ggplot()`. 
# MAGIC 
# MAGIC However, it is easier and faster to plot **R dataframe objects** in `ggplot()`. 
# MAGIC 
# MAGIC Thus, I recommend; 
# MAGIC 1. Reduce  a **tbl_spark  object** to key aggregate statistics (or a small set of data points) 
# MAGIC 2. Convert the stats into a **R dataframe object** 
# MAGIC 3. Plot it in `ggplot()`

# COMMAND ----------

# MAGIC %r
# MAGIC mtcars2 <- tbl_mtcars %>% 
# MAGIC   dplyr::mutate(  # do some manipulation if desired 
# MAGIC     num_cyl = cyl,
# MAGIC     cyl = as.character(cyl)
# MAGIC   ) %>% 
# MAGIC  dplyr::collect() # convert into a R dataframe: nearly the same thing as as.data.frame() 
# MAGIC 
# MAGIC class(mtcars2)

# COMMAND ----------

# MAGIC %r
# MAGIC library(dplyr) # note: loading SparkR and dplyr together may cause some namespace conflicts 
# MAGIC library(ggplot2)

# COMMAND ----------

# MAGIC %md
# MAGIC Now **plotting a R dataframe object is a standard use of ggplot**. 
# MAGIC 
# MAGIC When visualizing data for an analysis, one can almost always reduce it to key statistics. 
# MAGIC 
# MAGIC So, there is **no disadvantage** in converting a small dataset of stats into **a R dataframe**.
# MAGIC 
# MAGIC Once key stats are obtained, it is worth trying various ways to visualize them for the best visualization.
# MAGIC 
# MAGIC Making a plot into **a self-evident, visual story-telling material** usually takes refinements.      

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.2 ggplot2 for data visualization 
# MAGIC 
# MAGIC **dplyr** (for data wrangling) + **ggplot2** (for data plotting) are very popular data science tools for R as a part of <a href="https://ggplot2.tidyverse.org/" target="_blank">tidyverse</a>. 
# MAGIC 
# MAGIC Basic syntax is simple and we demonstrate some in this notebook. 
# MAGIC  
# MAGIC Here are additional resources to browse through: 
# MAGIC - <a href="https://r4ds.had.co.nz/" target="_blank">R for Data Science</a>
# MAGIC - <a href="https://moderndive.com/" target="_blank">Modern Dive</a>
# MAGIC - <a href="https://psyteachr.github.io/" target="_blank">PsyTeachR</a> 
# MAGIC - <a href="https://englelab.gatech.edu/useRguide/" target="_blank">EngleLab</a>
# MAGIC - <a href="https://rkabacoff.github.io/datavis/" target="_blank">Data Visualization with R</a>
# MAGIC - <a href="https://socviz.co/index.html" target="_blank">Data Visualization a practical introduction</a>
# MAGIC - <a href="https://geocompr.robinlovelace.net/" target="_blank">Geocomputation with R</a>
# MAGIC 
# MAGIC Another popular tool is <a href="https://plotly.com/graphing-libraries/" target="_blank">plotly</a>. Its APIs are available for various programming languages, including R and python.   

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # add a layer of linear regression model fit for each cylinder type
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg, color = cyl)) +
# MAGIC     geom_point(aes(shape = cyl)) +
# MAGIC     geom_smooth(method = lm)
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # add a layer of smooth regression fit (locally estimated scatterplot smoothing: loess) across all cylinder types
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     geom_smooth()
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # add a layer of large yellow dots to indicate automatic transmission  
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(data = filter(mtcars, am == 0), color = "yellow", size = 5) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     geom_smooth() 
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC # add a character variable for transimission type 
# MAGIC mtcars2 <- mtcars2 %>% 
# MAGIC   mutate(am_ch = recode(am, "0" = "automatic", "1" = "manual"))
# MAGIC 
# MAGIC plotprint(
# MAGIC # plot subsets of data by transmission type
# MAGIC ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC   geom_point(aes(shape = cyl, color = cyl)) + 
# MAGIC   facet_wrap( ~ am_ch),
# MAGIC   width=8, height=3.5, just="top", x=4.0, y=8.0 
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC #  plot subsets of data by transmission type and number of gears 
# MAGIC ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC   geom_point(aes(shape = cyl, color = cyl)) + 
# MAGIC   facet_grid(gear ~ am_ch),
# MAGIC   width=8, height=5.5, just="top", x=4.0, y=8.0 
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # change the displayed values on the y axis 
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     scale_y_continuous(breaks = seq(10, 36, by = 4))
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # map in log10 scale 
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     scale_x_log10() + scale_y_log10() 
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # change theme to black and white and overwrite axis labels 
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     theme_bw() + labs(x = "Horse power", y = "Miles per gallon")
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # overwrite the *joint legend* for color and shape attributes
# MAGIC   ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC     geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC     guides(
# MAGIC       color = guide_legend(title ="cylinder", override.aes = list(size = 4)),
# MAGIC       shape = guide_legend(title ="cylinder", override.aes = list(size = 4))
# MAGIC       )
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC library(ggrepel)
# MAGIC mtcars2$car_model <- rownames(mtcars)
# MAGIC 
# MAGIC plotprint(
# MAGIC # add labels of car model for cars that have either hp > 200 or mpg > 25
# MAGIC ggplot(mtcars2, aes(x = hp, y = mpg)) +
# MAGIC   geom_point(aes(shape = cyl, color = cyl)) +
# MAGIC   ggrepel::geom_label_repel(aes(label = car_model),
# MAGIC                             data = filter(mtcars2, hp > 200 | mpg > 25))
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # example of boxplot 
# MAGIC   ggplot(mtcars2, aes(x = am_ch, y = wt)) +
# MAGIC     geom_boxplot() + 
# MAGIC     geom_label_repel(aes(label = car_model),
# MAGIC                      data = filter(mtcars2, wt > 4.5 | wt < 3, am == 0))
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # examples of histogram #1  
# MAGIC   ggplot(mtcars2, aes(x = wt, fill = am_ch)) + 
# MAGIC     geom_histogram(binwidth = .75)
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC   # examples of histogram #2  
# MAGIC   ggplot(mtcars2, aes(x = wt, color = am_ch)) + 
# MAGIC     geom_freqpoly(binwidth = .75, position="dodge", size = 2)
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC # examples of barplot #1
# MAGIC ggplot(mtcars2, aes(x = cyl, fill = am_ch)) + geom_bar()
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC # examples of barplot #2
# MAGIC ggplot(mtcars2, aes(x = cyl, fill = am_ch)) + geom_bar(position = "dodge")
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC plotprint(
# MAGIC # examples of barplot #3
# MAGIC ggplot(mtcars2, aes(x = cyl, fill = am_ch)) + geom_bar(position = "fill") + labs(y = "fraction") 
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Python data analysis and visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Interacting with Spark

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1.1 spark.read

# COMMAND ----------

df_people = spark.read.format('delta').load('dbfs:/user/hive/warehouse/people10m')
display(df_people.limit(100))

# COMMAND ----------

display(df_people.dtypes)

# COMMAND ----------

import pandas as pd 
from pyspark.sql.functions import year

display(
    df_people
    .select('gender',year('birthDate').alias('birth_year'))  
    .crosstab('birth_year','gender')
) 

# COMMAND ----------

from pyspark.sql.functions import mean, round, min, max 

display(
    df_people
    .groupBy('gender')
    .agg(
        round(mean('salary'),0).alias('meanSalary'),
        min('salary').alias('minSalary'),
        max('salary').alias('maxSalary')
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 data visualization 
# MAGIC 
# MAGIC When we reduce data into key statistics or subsets and convert them into **Pandas dataframe**, various python data visualization tools are available. 
# MAGIC 
# MAGIC Here we will quickly touch on: 
# MAGIC - plotline: a near copy of ggplot (?) 
# MAGIC - seaborn: some functions are similar to ggplot
# MAGIC - matplotlib: from basic to highly customized plot
# MAGIC - plotly: very flexible for refining presentable figures 

# COMMAND ----------

# find data to use from storage 
display(dbutils.fs.ls('user/hive/warehouse/mtcars'))

# COMMAND ----------

# find which files to read
files = dbutils.fs.ls('user/hive/warehouse/mtcars')
files_parquet = [ file.path for file in files  if file.path.endswith('.parquet')]
files_parquet

# COMMAND ----------

# in this case there is only one, so it's easy to read 
df_mtcars = spark.read.format('parquet').load('dbfs:/user/hive/warehouse/mtcars/part-00000*')
display(df_mtcars)

# COMMAND ----------

mtcars = df_mtcars.toPandas()
print(type(mtcars))
mtcars.head()

# COMMAND ----------

# add character-type cylinder variable
mtcars['cyl_ch'] = mtcars['cyl'].astype(str)
mtcars['am_ch'] = mtcars['am'].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.1 plotnine
# MAGIC 
# MAGIC Popular **ggplot** in R is now available in python with having an almost identical syntax! 
# MAGIC - <a href="https://towardsdatascience.com/how-to-use-ggplot2-in-python-74ab8adec129"  target="_blank">how to use ggplot2 in python</a>
# MAGIC - <a href="https://realpython.com/ggplot-python/" target="_blank">RealPython tutorial</a>
# MAGIC - <a href="https://plotnine.readthedocs.io/en/stable/" target="_blank">plotnine doc</a> 

# COMMAND ----------

import pandas as pd 
from plotnine import *

ggplot(mtcars, aes(x='hp',y='mpg',color='cyl_ch')) + geom_point()

# COMMAND ----------

( ggplot(mtcars, aes(x='hp',y='mpg',color='cyl_ch')) 
 + geom_point()
 +  geom_smooth(method = 'lm')
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.2 seaborn 
# MAGIC 
# MAGIC <a href="https://seaborn.pydata.org/" target="_blank">seaborn</a> is also somewhat similar to ggplot. 

# COMMAND ----------

import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

warnings.filterwarnings("ignore")
color3 = sns.color_palette("hls", 3)

# Colored by Year
plt.figure(figsize=(8,6))  
sns.scatterplot('hp','mpg', data=mtcars, hue = 'cyl_ch', palette=color3);

# COMMAND ----------

g = (sns.lmplot('hp','mpg', data=mtcars, hue = 'cyl_ch', 
                palette=color3, legend=False) 
     .set(xlabel='Horsepower', ylabel='MPG'))

plt.legend(title='Cylinders', loc='upper right')
plt.show(g);

# COMMAND ----------

sns.relplot('hp','mpg', data=mtcars, hue = 'cyl_ch', palette=color3,
            row='am_ch', col = 'gear') 

# COMMAND ----------

mtcars['maker'] = mtcars['car_model'].apply(lambda x: x.split()[0])
top7_makers = mtcars['maker'].value_counts()[:7].index
mtcars_top7 = mtcars.set_index('maker').loc[top7_makers] 
mtcars_top7 

# COMMAND ----------

# example: barplot
(pd.crosstab(mtcars_top7.index, mtcars_top7.cyl_ch)
 .reindex(top7_makers) # order by the total counts
 .iloc[::-1] # reverse the order 
).plot.barh(stacked=True)
plt.xlabel('Models')
plt.legend(title = 'Cylinders')
plt.title('Cylinder Composition of Top Car Makers');

# COMMAND ----------

# example: boxplot
sns.boxplot(x = 'maker', y = 'mpg', hue = 'maker', data= mtcars_top7.reset_index())
plt.legend(loc = (1.04, 0.4))
plt.title("Model MPG distribution of top car makers");

# COMMAND ----------

df_people.limit(1000).display()

# COMMAND ----------

from pyspark.sql.functions import col

# convert a subset of spark dataframe to pandas dataframe
df_people_mod10 = (df_people 
    .withColumn("mod_10", col("id") % 10)   
    .filter(col('mod_10')==0))  # keep rows with id number is a multiple of 10 

df_people_mod10.limit(100).display()

pd_people = df_people_mod10.toPandas() 

# COMMAND ----------

pd_people['birth_year'] = pd_people['birthDate'].dt.year  # add extracted year 

# stats on salary by gender and birth year
df_gender_salary = (pd_people.groupby(['gender','birth_year'])['salary']
                    .agg(['count', np.mean, 'max'])
                    .rename(columns={'mean':'avg_salary', 'max':'max_salary'})
                    .reset_index())  

# plot mean salary along birth year by gender -- in this data salary doesn't depend on age or gender 
# for more interesting applications, use this type of graph mostly for time-series data
sns.relplot(x='birth_year',y='avg_salary',hue='gender',kind="line",
            palette=["#e74c3c", "#3498db"], 
            data=df_gender_salary[df_gender_salary['count'] > 500], 
            aspect=2); 

# COMMAND ----------

(sns.distplot(pd_people['salary'][pd_people['birth_year']==1983], bins=30)
 .set_title("Distribution of Salary, birth year = 1983"));

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.3 Matplotlib

# COMMAND ----------

import matplotlib as mpl
import matplotlib.pyplot as plt

# COMMAND ----------

df_M_salary = df_gender_salary[(df_gender_salary['count'] > 500) & (df_gender_salary['gender']=='M')]
df_F_salary = df_gender_salary[(df_gender_salary['count'] > 500) & (df_gender_salary['gender']=='F')]

plt.plot(df_M_salary['birth_year'], df_M_salary['avg_salary'], label='Male')
plt.plot(df_F_salary['birth_year'], df_F_salary['avg_salary'], label='Female')
plt.title('Average Salary by Birth Year and Gender')
plt.xlabel("Birth Year")
plt.ylabel("Average Salary")
axes=plt.gca()
axes.legend()
axes.set_aspect(0.025)
plt.show();

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 4.2.4 Plotly

# COMMAND ----------

import plotly as py 
import plotly.graph_objs as go

# COMMAND ----------

# let's do some data prep to feed into a customized plotly graph

df_gender_salary2 = df_gender_salary[df_gender_salary['count'] > 500] 
min_birth_year = df_gender_salary2['birth_year'].min()
max_birth_year = df_gender_salary2['birth_year'].max()

# Normalize at the value of the min birth year as the baseline 
df_gender_salary2 = df_gender_salary2.assign(
    _baseline = (df_gender_salary2['birth_year']==min_birth_year) * df_gender_salary2['avg_salary']) # get salary of base birth_year

df_gender_salary2['baseline'] = (df_gender_salary2.groupby('gender')['_baseline']).transform(max) # distribute across all rows
df_gender_salary2['salary_to_base'] = df_gender_salary2['avg_salary']/df_gender_salary2['baseline']

df_gender_salary2.head()

# COMMAND ----------

# pivot dataframe to be in a "wide" shape
df_gender_salary_wide = df_gender_salary2 \
    .pivot(index='birth_year', columns = 'gender', values=['avg_salary','salary_to_base'])

df_gender_salary_wide.head()

# COMMAND ----------

# after pivot, index and column structures change, so let's fix column names for our purpose
colnames1 = []
for i in range(len(df_gender_salary_wide.columns.get_level_values(0))):
    colnames1.append(df_gender_salary_wide.columns.get_level_values(0)[i] + '_' 
                     + df_gender_salary_wide.columns.get_level_values(1)[i])
    
df_gender_salary_wide.columns = colnames1 
df_gender_salary_wide.head()

# COMMAND ----------

# some final prep for a custome plot of: baseline-normalized line plot with end-point labels

y_vars = ['salary_to_base_F','salary_to_base_M']
labels = ['Female', 'Male']
colors = ['#e74c3c ', '#3498db']
fig_title = 'Fig. Average Salary Growth by Gender and Birth Year'
yaxis_title=f"Annual Salary, base = (birth year={min_birth_year})"
annotation_txt = 'Data source: People10m. sample data'    

x_data = np.arange(min_birth_year, max_birth_year+1)
y_data = df_gender_salary_wide[y_vars].sort_index()

fct_resize = .75
mode_size = [12, 12]
line_size = [5, 5]

# COMMAND ----------

# see https://plot.ly/python/line-charts/

fig = go.Figure()

for i in range(0, y_data.shape[1]):
    fig.add_trace(go.Scatter(x=x_data, y=y_data.iloc[:,i], mode='lines',
        name=labels[i],
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
    ))

    # endpoints
    fig.add_trace(go.Scatter(
        x=[x_data[-1]],
        y=[y_data.iloc[-1,i]],
        mode='markers',
        marker=dict(color=colors[i], size=mode_size[i])
    ))
    
fig.update_layout(
    height= 600 * fct_resize,
    width = 1000 * fct_resize,
    title = fig_title,
    yaxis_title=yaxis_title,
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2 * fct_resize,
        ticks='outside',
        tickfont=dict(
            #family='Arial',
            size=15 * fct_resize,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=True,
        showline=True,
        linecolor='rgb(204, 204, 204)',
        zerolinecolor='#D3D3D3',
        showticklabels=True,
        ticks='outside',
        tickfont=dict(
            #family='Arial',
            size=15 * fct_resize,
            color='rgb(82, 82, 82)',
        ),
    ),
    font=dict(size=15 * fct_resize),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=60,
        r=150,
        t=40,
    ),
    showlegend=False,
    plot_bgcolor='white',
)

annotations = []

annotations.append(
    dict(xref='paper', yref='paper',
        x= 1, #x= 0.02,
        xanchor= 'right',
        y= 1, #y=-.12,
        yanchor= 'bottom',
        showarrow=False,
        text= annotation_txt))

for y_trace, label, color in zip(y_data.to_numpy()[-1], labels, colors):
    
    # labeling the right_side of the plot
    annotations.append(
        dict(xref='paper', x=0.95, y=y_trace,
              xanchor='left', yanchor='middle',
              text='{:-.2f}%'.format(y_trace*100-100) + ' ' + label,
              font=dict(#family='Arial',
                        size=16 * fct_resize),
              showarrow=False))
        
fig.update_layout(annotations=annotations)    
fig.show()

# COMMAND ----------

sp_car = spark.read.csv('/databricks-datasets/Rdatasets/data-001/csv/MASS/Cars93.csv', header=True)

# COMMAND ----------

pd_car = sp_car.toPandas()

# COMMAND ----------

pd_car.head()

# COMMAND ----------

cnt_by_manuf = pd_car['Manufacturer'].value_counts() 
major_manuf = cnt_by_manuf[cnt_by_manuf >= 3]
major_manuf

# COMMAND ----------

pd_car['MPG.city'] = pd_car['MPG.city'].astype(float)
pd_car['MPG.highway'] = pd_car['MPG.highway'].astype(float)
mpg_stat = pd_car.groupby('Manufacturer')[['MPG.city','MPG.highway']] \
    .agg(np.mean).loc[major_manuf.index].sort_values(by='MPG.highway',ascending=True) 
mpg_stat 

# COMMAND ----------

data = mpg_stat 

var1 ="MPG.city"
name1="MPG City"
var2 ="MPG.highway"
name2="MPG Highway"

fig_title="Manufacturer Average MPG"
xaxis_title="Miles per gallon"
annotation_text= 'Data source: MASS Cars93.'

# COMMAND ----------

fct_resize = .75 

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data[var1],
    y=data.index,
    marker=dict(color="#2593ff", size=15 * fct_resize),
    mode="markers",
    name = name1
))

fig.add_trace(go.Scatter(
    x=data[var2],
    y=data.index,
    marker=dict(color="#ff8e25", size=15 * fct_resize),
    mode="markers",
    name=name2,
))

fig.update_layout(title=fig_title,
                  xaxis_title=xaxis_title,
                  yaxis_title="",
                    width=800 * fct_resize,
                    height=600 * fct_resize,
                    margin=dict(l=40, r=40, b=40, t=40),
                    font=dict(size=15 * fct_resize),
                    legend=dict(
                        font_size=14 * fct_resize,
                        x = 1,
                        y = 0.1,
                        yanchor='bottom',
                        xanchor='right',
                    ),
                   #paper_bgcolor='white',
                     plot_bgcolor='white',
                      annotations=[go.layout.Annotation(
                        xref='paper',
                        yref='paper',
                        x= 1, #x= 0.02,
                        xanchor= 'right',
                        y= 1, #y=-.12,
                        yanchor= 'bottom',
                        showarrow=False,
                        text= annotation_text),]
                 )
fig.update_xaxes(showline=False, #linewidth=2, linecolor= '#D3D3D3', 
                 showgrid=True, gridwidth=1, gridcolor= '#D3D3D3',
                 zeroline=True, zerolinewidth=2, zerolinecolor='#D3D3D3')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor= 'slategray')

fig.show()

# COMMAND ----------



# COMMAND ----------


