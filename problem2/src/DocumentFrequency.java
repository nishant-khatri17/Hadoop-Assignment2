import java.io.*;
import java.net.URI;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;

public class DocumentFrequency {

    public static class DFMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> stopWords = new HashSet<>();
        private PorterStemmer stemmer = new PorterStemmer();
        private final static IntWritable one = new IntWritable(1);
        private Text term = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            // Load stopwords from local path
            String stopwordsFile = context.getConfiguration().get("stopwords.file");
            if (stopwordsFile != null) {
                BufferedReader reader = new BufferedReader(new FileReader(stopwordsFile));
                String line;
                while ((line = reader.readLine()) != null) {
                    String word = line.trim().toLowerCase();
                    if (!word.isEmpty()) stopWords.add(word);
                }
                reader.close();
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            // Remove non-alphabetic characters
            line = line.replaceAll("[^a-z\\s]", " ");
            String[] tokens = line.split("\\s+");

            // Use a set to count each term ONCE per document
            // (DF = number of documents, not occurrences)
            Set<String> seenTerms = new HashSet<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.length() > 2 && !stopWords.contains(token)) {
                    // Apply Porter Stemmer
                    String stemmed = stemmer.stem(token);
                    if (!stemmed.isEmpty() && !seenTerms.contains(stemmed)) {
                        seenTerms.add(stemmed);
                        term.set(stemmed);
                        context.write(term, one);
                    }
                }
            }
        }
    }

    public static class DFReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem2/stopwords.txt");

        Job job = Job.getInstance(conf, "document frequency");
        job.setJarByClass(DocumentFrequency.class);
        job.setMapperClass(DFMapper.class);
        job.setCombinerClass(DFReducer.class);
        job.setReducerClass(DFReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
