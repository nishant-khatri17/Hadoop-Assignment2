import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class CoOccurrencePairsMapFn {

    public static class PairsMapFnMapper extends Mapper<Object, Text, Text, IntWritable> {

        private Set<String> stopWords = new HashSet<>();
        private Set<String> topWords = new HashSet<>();
        private int distance = 1;
        private Text pair = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("cooccurrence.distance", 1);
            loadFile(conf.get("stopwords.file"), stopWords);
            loadFile(conf.get("topwords.file"), topWords);
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
                System.err.println("Error: " + ioe.getMessage());
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            line = line.replaceAll("[^a-z\\s]", " ");
            String[] tokens = line.split("\\s+");

            List<String> words = new ArrayList<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.length() > 2 && !stopWords.contains(token)
                        && topWords.contains(token)) {
                    words.add(token);
                }
            }

            // MAP-FUNCTION LEVEL AGGREGATION
            // Buffer counts within this single map() call
            Map<String, Integer> localBuffer = new HashMap<>();
            for (int i = 0; i < words.size(); i++) {
                for (int j = i + 1; j <= Math.min(i + distance, words.size() - 1); j++) {
                    String p1 = "(" + words.get(i) + "," + words.get(j) + ")";
                    String p2 = "(" + words.get(j) + "," + words.get(i) + ")";
                    localBuffer.merge(p1, 1, Integer::sum);
                    localBuffer.merge(p2, 1, Integer::sum);
                }
            }

            // Emit aggregated counts for this line
            for (Map.Entry<String, Integer> entry : localBuffer.entrySet()) {
                pair.set(entry.getKey());
                context.write(pair, new IntWritable(entry.getValue()));
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) sum += val.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        int d = 1;
        String inputPath = args[0];
        String outputPath = args[1];
        for (int i = 2; i < args.length; i++) {
            if ("-d".equals(args[i])) d = Integer.parseInt(args[++i]);
        }
        conf.setInt("cooccurrence.distance", d);
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/stopwords.txt");
        conf.set("topwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/top50words_only.txt");

        Job job = Job.getInstance(conf, "pairs mapfn aggregation d=" + d);
        job.setJarByClass(CoOccurrencePairsMapFn.class);
        job.setMapperClass(PairsMapFnMapper.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
