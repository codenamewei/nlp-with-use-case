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

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/**
 *  This is a DataSetIterator customized for use case of spam mail filtering.
 *  It takes
 *  (1) train or test data directory (through variable boolean train)
 *      Each directory above contains two subdirectories of spam and non-spam.
 *  (2) Word vectors embeddings
 *      TGoogle news vector is used and it can be downloaded through https://code.google.com/p/word2vec/
 *      It contains pretrained word embeddings with length of 300
 *
 *  Labels/target: two classes, non-spam(negative) or spam(positive), predicted at the final time step (word) of each mail.
 *
 *  @author ChiaWei Lim
 */
public class SpamMailDataSetIterator implements DataSetIterator
{
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int spamCursor = 0;
    private int nonSpamCursor = 0;
    private double spamNonSpamRatio;

    private final File[] spamFiles;
    private final File[] nonSpamFiles;
    private final TokenizerFactory tokenizerFactory;

    /**
     * @param dataRootDirectory the directory of the Spam Mail Dataset
     * @param wordVectors WordVectors object
     * @param batchSize Size of each minibatch for training
     * @param truncateLength If mail text exceed
     * @param train If true: return the training data. If false: return the testing data.
     */
    public SpamMailDataSetIterator(String dataRootDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException
    {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

        //Training or testing dataset
        String subDirectory = (train) ? "/train/" : "/test/";

        String dataDirectory = dataRootDirectory + subDirectory;

        //Get text files in both spam and non spam folder
        //Spam mail as positive label, normal (non-spam) mail as negative label
        File pos = new File(dataDirectory , "spam/");
        File neg = new File(dataDirectory , "non-spam/");

        spamFiles = pos.listFiles();
        nonSpamFiles = neg.listFiles();

        this.spamNonSpamRatio = spamFiles.length / (double) nonSpamFiles.length;

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num)
    {
        if((spamCursor == spamFiles.length) && nonSpamCursor == nonSpamFiles.length)
        {
            throw new NoSuchElementException();
        }
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int batchSize) throws IOException
    {
        //Load mails to String. Unbalanced dataset
        //Include data points of both labels in a ratio proportion
        List<String> mailArray = new ArrayList<>(batchSize);
        boolean[] positive = new boolean[batchSize];

        int posBatchSize = 0;
        if(spamCursor < spamFiles.length)
        {
            posBatchSize = (int) Math.floor(spamNonSpamRatio * batchSize);

            for( int i = 0; i < posBatchSize  && spamCursor < spamFiles.length; ++i)
            {
                String mail = FileUtils.readFileToString(spamFiles[spamCursor], "UTF-8");
                mailArray.add(mail);
                positive[i] = true;
                ++spamCursor;
            }
        }

        //negative index starts after positive index
        for( int i = posBatchSize; i< batchSize && nonSpamCursor < nonSpamFiles.length; ++i )
        {
            //load non-spam data
            String mail = FileUtils.readFileToString(nonSpamFiles[nonSpamCursor], "UTF-8");
            mailArray.add(mail);
            positive[i] = false;
            ++nonSpamCursor;
        }

        //Tokenize mails and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(mailArray.size());
        int maxLength = 0;
        for(String s : mailArray){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest mail exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have mailArray.size() examples of varying lengths
        INDArray features = Nd4j.create(mailArray.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(mailArray.size(), 2, maxLength);    //Two labels: positive or negative

        //Padding arrays because of mails of different lengths and only one output at the final time step
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(mailArray.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(mailArray.size(), maxLength);

        for( int i = 0; i < mailArray.size(); ++i)
        {
            List<String> tokens = allTokens.get(i);

            // Get the truncated sequence length of document (i)
            int seqLength = Math.min(tokens.size(), maxLength);

            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; ++j){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
                featuresMask.get(new INDArrayIndex[] {NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)}).assign(1);
            }

            int idx = (positive[i] ? 1 : 0);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [1, 0] for negative, [0, 1] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    public int totalExamples() {
        return spamFiles.length + nonSpamFiles.length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        spamCursor = 0;
        nonSpamCursor = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("non-spam","spam");
    }

    @Override
    public boolean hasNext() {
        return (spamCursor + nonSpamCursor) < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

        throw new UnsupportedOperationException();
    }

    @Override
    public  DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Used post training to load a mail from a file to a features INDArray that can be passed to the network output method
     *
     * @param file      File to load the mail text from
     * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not truncate
     * @return          Features array
     * @throws IOException If file cannot be read
     */
    public INDArray loadFeaturesFromFile(File file, int maxLength) throws IOException {
        String mail = FileUtils.readFileToString(file, "UTF-8");
        return loadFeaturesFromString(mail, maxLength);
    }

    /**
     * Used post training to convert a String to a features INDArray that can be passed to the network output method
     *
     * @param mailContents Contents of the mail to vectorize
     * @param maxLength Maximum length (if mail is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not truncate
     * @return Features array for the given input String
     */
    public INDArray loadFeaturesFromString(String mailContents, int maxLength)
    {
        List<String> tokens = tokenizerFactory.create(mailContents).getTokens();
        List<String> tokensFiltered = new ArrayList<>();

        for(String t : tokens){
            if(wordVectors.hasWord(t)) tokensFiltered.add(t);
        }
        int outputLength = Math.min(maxLength,tokensFiltered.size());

        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        int count = 0;
        for( int j=0; j < tokensFiltered.size() && count < maxLength; ++j)
        {
            String token = tokensFiltered.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            if(vector == null) continue;   //Word not in word vectors

            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
            ++count;
        }

        return features;
    }
}