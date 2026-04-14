import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class CoOccurrenceStripesMapFn {

    public static class StripesMapFnMapper extends Mapper<Object, Text, Text, MapWritable> {

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

            // MAP-FUNCTION LEVEL: buffer within this single map() call
            Map<String, MapWritable> localBuffer = new HashMap<>();
            for (int i = 0; i < words.size(); i++) {
                String w = words.get(i);
                if (!localBuffer.containsKey(w)) {
                    localBuffer.put(w, new MapWritable());
                }
                for (int j = Math.max(0, i - distance); j <= Math.min(i + distance, words.size() - 1); j++) {
                    if (i == j) continue;
                    Text neighbor = new Text(words.get(j));
                    MapWritable stripe = localBuffer.get(w);
                    if (stripe.containsKey(neighbor)) {
                        ((IntWritable) stripe.get(neighbor)).set(((IntWritable) stripe.get(neighbor)).get() + 1);
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
                    }
                }
            }

            // Emit aggregated stripes for this line
            for (Map.Entry<String, MapWritable> entry : localBuffer.entrySet()) {
                wordKey.set(entry.getKey());
                context.write(wordKey, entry.getValue());
            }
        }
    }

    public static class StripesReducer extends Reducer<Text, MapWritable, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {
            Map<String, Integer> merged = new TreeMap<>();
            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    String neighbor = entry.getKey().toString();
                    int count = ((IntWritable) entry.getValue()).get();
                    merged.merge(neighbor, count, Integer::sum);
                }
            }
            StringBuilder sb = new StringBuilder("{");
            for (Map.Entry<String, Integer> entry : merged.entrySet()) {
                sb.append(entry.getKey()).append(":").append(entry.getValue()).append(", ");
            }
            if (sb.length() > 1) sb.setLength(sb.length() - 2);
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
            if ("-d".equals(args[i])) d = Integer.parseInt(args[++i]);
        }
        conf.setInt("cooccurrence.distance", d);
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/stopwords.txt");
        conf.set("topwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem1/top50words_only.txt");

        Job job = Job.getInstance(conf, "stripes mapfn aggregation d=" + d);
        job.setJarByClass(CoOccurrenceStripesMapFn.class);
        job.setMapperClass(StripesMapFnMapper.class);
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
