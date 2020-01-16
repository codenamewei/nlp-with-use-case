/* *****************************************************************************
Copyright 2019 codenamewei

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
******************************************************************************/

package ai.codenamewei;

import akka.japi.Pair;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Word2vec implementation in DL4J using Skip Gram
 *
 * Data can be found in https://drive.google.com/drive/u/0/folders/1zcLtMEKvYN2mM1ykRQr1EP1IHlZ-z3ng
 * @author codenamewei
 */
@Slf4j
public class SkipGramImp
{
    public static void main(String[] args) throws Exception
    {
        String rootDir = System.getProperty("java.io.tmpdir");
        String modelSavedName = "word2vec.zip";

        File modelSavedPath = new File(rootDir + "\\" + modelSavedName);

        Word2Vec word2vec = null;

        //Train model
        if(!modelSavedPath.exists())
        {
            File dataPath = new ClassPathResource("hotel-raw-data.csv").getFile();

            log.info("Load & Vectorize Sentences...");

            //Get each sentence, split the lines
            SentenceIterator dataIter = new BasicLineIterator(dataPath);
            dataIter.setPreProcessor(new SentencePreProcessor() {
                @Override
                public String preProcess(String sentence)
                {
                    if( sentence.split(" ").length < 3) //remove sentence with length fewer than 2
                    {
                        return "";
                    }

                    return sentence.toLowerCase();
                }
            });


            //Get Words
            TokenizerFactory tokenizer = new DefaultTokenizerFactory();

            /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
            */
            tokenizer.setTokenPreProcessor(new CommonPreprocessor());

            log.info("Building model...");
            int layerSize = 80; //Word Dimensionality: Conventional in mikolov paper is 50 - 100
            int minWordFrequency = 10;
            int seed = 123;
            int windowSize = 5; //Paper written 5
            int batchSize = 1024; //default when undefined is 512

            //stop words = useless words / most common words such as: a, an, the...
            //retrieved from NLTK
            String[] stopWordsArray = {"no", "negative", "ourselves", "hers", "between", "yourself", "but", "again", "there",
                    "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some",
                    "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s",
                    "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we",
                    "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down",
                    "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when",
                    "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then",
                    "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has",
                    "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs",
                    "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"};

            List<String> stopWords = Arrays.asList(stopWordsArray);

            word2vec = new Word2Vec.Builder()
                    .epochs(1)
                    .iterate(dataIter)
                    .layerSize(layerSize)
                    .useAdaGrad(true)
                    .minWordFrequency(minWordFrequency)
                    .seed(seed)
                    .windowSize(windowSize)
                    .tokenizerFactory(tokenizer)
                    .stopWords(stopWords)
                    .batchSize(batchSize)
                    .build();

            log.info("Fitting Word2Vec model");
            word2vec.fit();

            log.info("Save Word2Vec Model as {}", modelSavedPath);
            WordVectorSerializer.writeWord2VecModel(word2vec, modelSavedPath);


        }
        else //Load model
        {
            word2vec = WordVectorSerializer.readWord2VecModel(modelSavedPath);
        }


        String testWord = "breakfast";
        String testWord2 = "room";

        INDArray vector = word2vec.getWordVectorMatrix(testWord);
        log.info("\n");
        log.info("Word vector length: " + vector.columns() + "\n"); //determined by layerSize


        sanityCheck(word2vec, testWord);
        sanityCheck(word2vec, testWord2);


        Pair<String, String> pair1 = new Pair<>("couple", "vacation");
        Pair<String, String> pair2 = new Pair<>("couple", "business");

        double dist1 = compareDistanceBetweenWords(word2vec, pair1);
        double dist2 = compareDistanceBetweenWords(word2vec, pair2);

        log.info("Dist between {}: {}\n", pair1, dist1);
        log.info("Dist between {}: {}\n", pair2, dist2);

    }

    public static void sanityCheck(Word2Vec model, String word) throws Exception {

        int numWordsNearest = 10;

        if (!model.hasWord(word)) {
            throw new Exception("Word trained with not found!");
        }

        //wordsNearestSum have the absense of mean compared to wordsNearest
        //Another words, means no positive and negative, but rather a measure of # number of nearest words
        Collection<String> list = model.wordsNearestSum(word, numWordsNearest + 1);
        list.remove(word); // the first word is the word itself

        log.info("10 Words closest to {}: {}\n", word, list);

    }

    /**
     * Compare distance between pair of words
     */
    public static double compareDistanceBetweenWords(Word2Vec model, Pair<String, String> pair)
    {
        //Option: INDArray getWordVectorMatrix

        /*
        double[] matrix1 = model.getWordVector(pair.first());
        double[] matrix2 = model.getWordVector(pair.second());
        return cosineSimilarity(matrix1, matrix2);
        */

        return model.similarity(pair.first(), pair.second());
    }


    /*
    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    */

}