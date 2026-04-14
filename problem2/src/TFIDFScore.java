import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;

public class TFIDFScore {

    public static class TFIDFMapper extends Mapper<Object, Text, Text, Text> {

        private Set<String> stopWords = new HashSet<>();
        private Map<String, Integer> dfMap = new HashMap<>();
        private PorterStemmer stemmer = new PorterStemmer();
        private String docId = "";

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();

            // Load stopwords
            String stopwordsFile = conf.get("stopwords.file");
            if (stopwordsFile != null) {
                BufferedReader reader = new BufferedReader(new FileReader(stopwordsFile));
                String line;
                while ((line = reader.readLine()) != null) {
                    String word = line.trim().toLowerCase();
                    if (!word.isEmpty()) stopWords.add(word);
                }
                reader.close();
            }

            // Load DF values from cached TSV file (TERM\tDF)
            String dfFile = conf.get("df.file");
            if (dfFile != null) {
                BufferedReader reader = new BufferedReader(new FileReader(dfFile));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\t");
                    if (parts.length == 2) {
                        try {
                            dfMap.put(parts[0].trim(), Integer.parseInt(parts[1].trim()));
                        } catch (NumberFormatException e) {
                            // skip malformed lines
                        }
                    }
                }
                reader.close();
            }
            System.err.println("Loaded " + dfMap.size() + " DF terms");
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            // Use filename as document ID
            String filename = ((org.apache.hadoop.mapreduce.lib.input.FileSplit)
                context.getInputSplit()).getPath().getName();
            docId = filename.replace(".txt", "");

            String line = value.toString().toLowerCase();
            line = line.replaceAll("[^a-z\\s]", " ");
            String[] tokens = line.split("\\s+");

            // Count TF for each stemmed term in this document
            Map<String, Integer> tfMap = new HashMap<>();
            for (String token : tokens) {
                token = token.trim();
                if (token.length() > 2 && !stopWords.contains(token)) {
                    String stemmed = stemmer.stem(token);
                    if (dfMap.containsKey(stemmed)) {
                        tfMap.merge(stemmed, 1, Integer::sum);
                    }
                }
            }

            // Emit (docId, term\ttf) for each term
            for (Map.Entry<String, Integer> entry : tfMap.entrySet()) {
                String term = entry.getKey();
                int tf = entry.getValue();
                int df = dfMap.get(term);
                // SCORE = TF * log(10000/DF + 1)
                double score = tf * Math.log(10000.0 / df + 1);
                // Emit: key = docId, value = term\tscore
                context.write(new Text(docId),
                    new Text(term + "\t" + String.format("%.4f", score)));
            }
        }
    }

    public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            // key = docId, values = [term\tscore, ...]
            for (Text val : values) {
                String[] parts = val.toString().split("\t");
                if (parts.length == 2) {
                    // Output: ID\tTERM\tSCORE
                    context.write(key, new Text(parts[0] + "\t" + parts[1]));
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("stopwords.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem2/stopwords.txt");
        conf.set("df.file",
            "/Users/nishantkhatri/Desktop/hadoop/problem2/top100df.txt");

        Job job = Job.getInstance(conf, "tfidf score");
        job.setJarByClass(TFIDFScore.class);
        job.setMapperClass(TFIDFMapper.class);
        job.setReducerClass(TFIDFReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setInputFormatClass(CustomFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
