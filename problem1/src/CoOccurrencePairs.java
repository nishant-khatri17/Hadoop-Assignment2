import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class CoOccurrencePairs {

    public static class PairsMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> stopWords = new HashSet<>();
        private Set<String> topWords = new HashSet<>();
        private final static IntWritable one = new IntWritable(1);
        private Text pair = new Text();
        private int distance = 1;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("cooccurrence.distance", 1);

            // Read directly from local filesystem paths
            loadFile(conf.get("stopwords.file"), stopWords);
            loadFile(conf.get("topwords.file"), topWords);

            System.err.println("Loaded " + stopWords.size() + " stopwords");
            System.err.println("Loaded " + topWords.size() + " topwords");
        }

        private void loadFile(String fileName, Set<String> set) {
            if (fileName == null) return;
            try {
                BufferedReader reader = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = reader.readLine()) != null) {
                    String word = line.trim().toLowerCase();
                    if (!word.isEmpty()) set.add(word);
                }
                reader.close();
            } catch (IOException ioe) {
                System.err.println("Error reading file " + fileName + ": " + ioe.getMessage());
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            // Remove non-alphabetic characters
            line = line.replaceAll("[^a-z\\s]", " ");
            String[] tokens = line.split("\\s+");

            // Keep only top50 words that are not stopwords
            List<String> words = new ArrayList<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.length() > 2
                        && !stopWords.contains(token)
                        && topWords.contains(token)) {
                    words.add(token);
                }
            }

            // Emit co-occurring pairs within distance d
            for (int i = 0; i < words.size(); i++) {
                for (int j = i + 1; j <= Math.min(i + distance, words.size() - 1); j++) {
                    // Emit (wordA, wordB)
                    pair.set("(" + words.get(i) + "," + words.get(j) + ")");
                    context.write(pair, one);
                    // Emit (wordB, wordA)
                    pair.set("(" + words.get(j) + "," + words.get(i) + ")");
                    context.write(pair, one);
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
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

        // Default distance
        int d = 1;
        String inputPath = args[0];
        String outputPath = args[1];

        // Parse -d argument
        for (int i = 2; i < args.length; i++) {
            if ("-d".equals(args[i])) {
                d = Integer.parseInt(args[++i]);
            }
        }

        conf.setInt("cooccurrence.distance", d);

        // Set local file paths directly
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/stopwords.txt");
        conf.set("topwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/top50words_only.txt");

        Job job = Job.getInstance(conf, "cooccurrence pairs d=" + d);
        job.setJarByClass(CoOccurrencePairs.class);
        job.setMapperClass(PairsMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Use instructor's custom input format
        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}