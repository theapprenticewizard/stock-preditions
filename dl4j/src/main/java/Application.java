import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class Application {

    public static void main(String[] args) {
        if (args.length > 0 && args[0] != null) {
            printPredictions(args[0]);
        }

        System.err.println("path not provided for file");
    }

    public static void printPredictions(String path) {
        try(InputStream is = new FileInputStream(path)) {


            SparkConf conf = new SparkConf();

            // set spark to run locally with `n` threads, where n is the amount of logical cores on the host CPU
            // alternatively for a single thread, you can run spark with a single core by running local[1] instead
            conf.setMaster("local[*]");
            conf.setAppName("Stock Price Predictor");

            // Allows native java to run using apparent native Collections but returns JavaRDD.
            JavaSparkContext sCtx = new JavaSparkContext(conf);

            // JavaRDD is a specific type for dealing with Resilient Distributed Datasets.
            JavaRDD<String> raw = sCtx.textFile(path);

            JavaRDD<List<Writable>> preProcessed = raw.map(new StringToWritablesFunction(new CSVLineSequenceRecordReader()));

            // execute the finished result
            JavaRDD<List<Writable>> processed = new SparkTransformExecutor().execute(preProcessed, createYahooFinanceTransformProcess());


            System.out.println(processed.toString());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Data Format:
     *
     * Date,Open,High,Low,Close,Adj Close,Volume
     * 2013-08-26,432.189026,434.623199,430.226776,430.395691,430.395691,2118600
     *
     * @return schema containing for yahoo finance stock history
     */
    private static Schema createYahooFinanceSchema() {
        return new Schema.Builder()
                .addColumnString("Date")
                .addColumnsDouble("Open", "High", "Low", "Close", "Adj Close")
                .addColumnLong("Volume")
                .build();
    }

    /**
     * Because we are training a `simple regression` we only need a single Y value, for this we have chosen
     * the `Close` column which is the closing value of the stock.  And as this is a time-series prediction
     * we need the input of the data to be of a date/time type in order for our machine learning model to
     * understand.
     *
     * @return A transformation that is Machine Learning ready,
     * this is step 2 of the 'Extract, Transform, Load` process
     */
    private static TransformProcess createYahooFinanceTransformProcess() {
        return new TransformProcess.Builder(createYahooFinanceSchema())
                .removeAllColumnsExceptFor("Date", "Close")
                .stringToTimeTransform("Date", "yyyy-mm-dd", DateTimeZone.forID("America/New_York"))
                .build();
    }
}
