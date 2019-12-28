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

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import lombok.extern.slf4j.Slf4j;

import java.io.File;

/**
 * This examples show spam mail filtering using a combination of Word2Vec vectors and LSTM.
 * Each mail text will be classified as either spam or non-spam based on the words it in.
 *
 *[CAVEAT] This example consume more memory than usual. Recommend to run on host more than 16GB.
 *
 * Default using of Word2Vec vectors in this example is Google News Word Vectors.
 * You would to download it first and change the WORD_VECTORS_PATH to the local directory path for GoogleNews-vectors-negative300.bin.gz.
 * The Google News vector model available here: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
 * The file is about 1.5GB
 *
 * You can customize to other pretrained word vectors.
 * A few things will have to be changed, file path to word vectors and length of word vector embeddings.
 *
 * The vectorization of the text is customized in SpamMailDataSetIterator.
 * The original dataset is unbalanced with more normal mail samples compared to spam mail samples. (Total: 5574, Spam: 4827 non-spam: 747)
 * This is a caveat where the network might adapt to patterns of non-spam mail better due to the volume of the data.
 * The performance may be better improved with a balanced dataset.
 *
 * The dataset is then preprocessed into a mail per file and segregated further into spam/ and non-spam/ for vectorization
 * The file directory is as below:
 *
 * resources/data
 * │
 * └───train
 * │   │───spam
 * │       │ 0.txt
 * │       │ 1.txt
 * │       │ ...
 * │   │───non-spam
 * │       │ 0.txt
 * │       │ 1.txt
 * │       │ ...
 * └───test
 * │   │───spam
 * │       │ 0.txt
 * │       │ 1.txt
 * │       │ ...
 * │   │───non-spam
 * │       │ 0.txt
 * │       │ 1.txt
 * │       │ ...
 *
 * The contribution of the dataset goes to http://dcomp.sor.ufscar.br/talmeida/smspamcollection/
 * The original dataset can be found in http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
 *
 * @author codenamewei
 */

@Slf4j
public class SpamMailFiltering
{
    // Google News Vector file path
    // Download the file and set this path manually before run the program.
    // Else, it will cause error
    // ------> PATH\TO\YOUR\VECTOR\GoogleNews-vectors-negative300.bin.gz

    public static final String WORD_VECTORS_PATH = "//Users//wei//Documents//models//";

    public static final String WORD_VECTOR_FILE = "GoogleNews-vectors-negative300.bin.gz";
    public static final int WORD_VECTORS_LENGTH = 300;  //Length of the word vectors. The length is 300 for Google News vector .

    public static final int SEED = 1234;                 //Seed for reproducibility
    public static final int EPOCHS = 1;                 //Full passes of training data
    public static final int CLASSES = 2;                //Number of classes, spam & normal mail
    public static final int BATCH_SIZE = 64;           //Number of examples in each minibatch
    public static final int TRUNCATED_LENGTH = 120;     //Truncate spam mail with length greater than this

    public static void main(String[] args) throws Exception
    {
        //Loading of pretrained word vectors
        log.info("******Loading pretrained embedding model. This would takes a while...******");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH + WORD_VECTOR_FILE));

        //Set data root directory path
        String dataBaseDir = new ClassPathResource("data").getFile().getAbsolutePath();

        //Data Loading into training and testing dataset
        SpamMailDataSetIterator trainIter = new SpamMailDataSetIterator(dataBaseDir, wordVectors, BATCH_SIZE, TRUNCATED_LENGTH, true);

        //Network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .updater(new Adam(5e-3))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)//clip gradient with value more than 1.0
                .list()

                .layer(new LSTM.Builder()
                        .nIn(WORD_VECTORS_LENGTH)
                        .nOut(TRUNCATED_LENGTH)
                        .activation(Activation.TANH)
                        .build())

                .layer(new RnnOutputLayer.Builder()
                        .nIn(TRUNCATED_LENGTH)
                        .nOut(CLASSES)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())

                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        //Visualization of training UI
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        model.setListeners(new StatsListener(statsStorage));

        //Train model
        log.info("Training model...");
        model.fit(trainIter, EPOCHS);

        //Evaluation on testing data set
        SpamMailDataSetIterator testIter = new SpamMailDataSetIterator(dataBaseDir, wordVectors, 1, TRUNCATED_LENGTH, false);

        log.info("Evaluating model...");
        Evaluation eval = model.evaluate(testIter);
        System.out.println(eval.confusionMatrix());

        //Loading of a spam mail for prediction
        String spamSample = "Congratulations! Call FREEFONE 08006344447 to claim your guaranteed £2000 CASH or £5000 gift. Redeem it now!";

        INDArray features = testIter.loadFeaturesFromString(spamSample, TRUNCATED_LENGTH);
        INDArray networkOutput = model.output(features);

        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\nSpam Mail Sample: \n" + spamSample + "\n");

        //Probability at last time step
        System.out.println("Probabilities as a Normal Mail: " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("Probabilities as a Spam Mail: " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("Program end...");
    }
}
