from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
from matplotlib import pyplot as plt


#Start session
spark = SparkSession.builder.master("local").appName("NYC Taxi Cab").getOrCreate()


#Create a custom schema for each column to load .csv faster
custom_schema = StructType([StructField("VendorID", IntegerType(), False),
						   StructField("tpep_pickup_datetime", TimestampType(), False),
                           StructField("tpep_dropoff_datetime", TimestampType(), False),
                           StructField("passenger_count", IntegerType(), False),
                           StructField("trip_distance", DoubleType(), False),
                           StructField("RatecodeID", IntegerType(), False),
                           StructField("store_and_fwd_flag", StringType(), False),
                           StructField("PULocationID", IntegerType(), False),
                           StructField("DOLocationID", IntegerType(), False),
                           StructField("payment_type", IntegerType(), False),
                           StructField("fare_amount", DoubleType(), False),
                           StructField("extra", DoubleType(), False),
                           StructField("mta_tax", DoubleType(), False),
                           StructField("tip_amount", DoubleType(), False),
                           StructField("tolls_amount", DoubleType(), False),
                           StructField("improvement_surcharge", DoubleType(), False),
                           StructField("total_amount", DoubleType(), False),
                           ])


#Load taxi cab data set
df = spark.read.csv("yellow_tripdata_2017-02.csv", header = True, schema = custom_schema, sep = ',')


#Create a new column called Duration, which will be the time difference (in minutes) between pickup and dropoff time
timeDiff = (unix_timestamp('tpep_dropoff_datetime')) - (unix_timestamp('tpep_pickup_datetime'))
df = df.withColumn("duration", timeDiff/60)

#Removes nosensical data  
df = df.where((df['fare_amount'] > 2.5) & (df['fare_amount'] < 100))        #Trips worth less than the base fare of $2.50 and more than $100
df = df.where((df['duration'] > 0) & (df['duration'] < 180))               #Negative Duration trips and trips longer than 2 hours
df = df.where((df['PULocationID'] < 264) & (df['DOLocationID'] < 264))     #Unknown pickup/dropoff locations
df = df.where(df['trip_distance'] < 20)                                    #Trips longer than 20 miles
df = df.where(df['passenger_count'] > 0)                                   #Trips with 0 or negative passengers
df = df.where(month('tpep_pickup_datetime') == 2)                          #Make sure pickup time is in the month of Feburary
df = df.where(df['payment_type'] < 3)                     		            #Remove miscellaneous payment types (Not cash or credit card)
df = df.where(df['RatecodeID'] == 1)                                        #RatecodeID only go from 1-6

#remove unused data columns
droplist = ['total_amount','tip_amount','payment_type','extra','mta_tax','improvement_surcharge',
             'tolls_amount','passenger_count','VendorID',"tpep_dropoff_datetime",'RatecodeID', 'duration']
df =df.drop(*droplist)

#pandas conversion doesn't seem to like timestamp types, so i'll convert them to string types
df = df.withColumn("tpep_pickup_datetime",  df["tpep_pickup_datetime"].cast(StringType()))


#count 4,884,385
df1 = df.where(df['fare_amount'] < 10)

#count 3,459,792
df2 = df.where((df['fare_amount'] >= 10) & (df['fare_amount'] <30))

#count 326,771
df3 = df.where((df['fare_amount'] >= 30) & (df['fare_amount'] <50))

#count 19,199
df4 = df.where(df['fare_amount'] >= 50)


df1 = df1.sample(withReplacement = False, fraction = 0.006)
df2 = df2.sample(withReplacement = False, fraction = 0.009)
df3 = df3.sample(withReplacement = False, fraction = 0.09)

df1 = df1.toPandas()
df2 = df2.toPandas()
df3 = df3.toPandas()
df4 = df4.toPandas()

df = pd.concat([df1, df2, df3, df4], ignore_index = True)

"""
# Reduce overall data to 88,617 datapoints. Too much data will take too long for tensorflow to converge
df = df.sample(withReplacement = False, fraction = 0.05)

"""

#Save new CSV file using Pandas so that file is saved as a single file
df.to_csv('cleaned_reduced_nyc_taxi_data.csv')

