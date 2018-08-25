import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.joda.time.DateTimeZone;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class Application {

    public static void main(String[] args) {
        if (args.length > 0 && args[0] != null) {
            printPredictions(args[0]);
        }

        System.err.println("path not provided for file");
    }

    public static void printPredictions(String path) {
        try(InputStream is = new FileInputStream(path)) {

            // load the file here

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
