import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class CoOccurrenceStripes {

    public static class StripesMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> stopWords = new HashSet<>();
        private Set<String> topWords = new HashSet<>();
        private int distance = 1;
        private Text wordKey = new Text();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            distance = conf.getInt("cooccurrence.distance", 1);
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
            line = line.replaceAll("[^a-z\\s]", " ");
            String[] tokens = line.split("\\s+");

            // Keep only top50 words
            List<String> words = new ArrayList<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.length() > 2
                        && !stopWords.contains(token)
                        && topWords.contains(token)) {
                    words.add(token);
                }
            }

            // Stripes: for each word emit a map of its neighbors
            for (int i = 0; i < words.size(); i++) {
                MapWritable stripe = new MapWritable();

                for (int j = i + 1; j <= Math.min(i + distance, words.size() - 1); j++) {
                    Text neighbor = new Text(words.get(j));
                    if (stripe.containsKey(neighbor)) {
                        IntWritable count = (IntWritable) stripe.get(neighbor);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
                    }
                }

                // Also count words before i
                for (int j = Math.max(0, i - distance); j < i; j++) {
                    Text neighbor = new Text(words.get(j));
                    if (stripe.containsKey(neighbor)) {
                        IntWritable count = (IntWritable) stripe.get(neighbor);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
                    }
                }

                if (!stripe.isEmpty()) {
                    wordKey.set(words.get(i));
                    context.write(wordKey, stripe);
                }
            }
        }
    }

    public static class StripesReducer extends Reducer<Text, MapWritable, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            // Merge all stripes for this word
            Map<String, Integer> merged = new TreeMap<>();
            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    String neighbor = entry.getKey().toString();
                    int count = ((IntWritable) entry.getValue()).get();
                    merged.merge(neighbor, count, Integer::sum);
                }
            }

            // Format output as: word {neighbor1:count, neighbor2:count, ...}
            StringBuilder sb = new StringBuilder("{");
            for (Map.Entry<String, Integer> entry : merged.entrySet()) {
                sb.append(entry.getKey()).append(":").append(entry.getValue()).append(", ");
            }
            if (sb.length() > 1) {
                sb.setLength(sb.length() - 2); // remove trailing ", "
            }
            sb.append("}");

            context.write(key, new Text(sb.toString()));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        int d = 1;
        String inputPath = args[0];
        String outputPath = args[1];

        for (int i = 2; i < args.length; i++) {
            if ("-d".equals(args[i])) {
                d = Integer.parseInt(args[++i]);
            }
        }

        conf.setInt("cooccurrence.distance", d);
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/stopwords.txt");
        conf.set("topwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/top50words_only.txt");

        Job job = Job.getInstance(conf, "cooccurrence stripes d=" + d);
        job.setJarByClass(CoOccurrenceStripes.class);
        job.setMapperClass(StripesMapper.class);
        job.setReducerClass(StripesReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
