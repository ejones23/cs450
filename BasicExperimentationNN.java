package neuralnetworkclassifier;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ejones23
 */
public class BasicExperimentationNN {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/iris.csv");
//        DataSource source = new DataSource("data/diabetes.arff");
        
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        Standardize standardizeFilter = new Standardize();
        standardizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, standardizeFilter);
        
        data.randomize(new Random(7));//Arbitrary seed
        
        String[] options = weka.core.Utils.splitOptions("-P 30");
        RemovePercentage remove = new RemovePercentage();
        remove.setOptions(options);
        remove.setInputFormat(data);
        Instances train = Filter.useFilter(data, remove);
        
        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        Instances test = Filter.useFilter(data, remove);
        
        int numNodesHL = 5;
        int increment = 50;
        int[] _1Hidden = {numNodesHL, data.numClasses()};
        int[] _2Hidden = {numNodesHL, numNodesHL, data.numClasses()};
        int[] _3Hidden = {numNodesHL, numNodesHL, numNodesHL, data.numClasses()};

        NeuralNetworkClassifier _1HC = new NeuralNetworkClassifier(_1Hidden, increment, train);
        NeuralNetworkClassifier _2HC = new NeuralNetworkClassifier(_2Hidden, increment, train);
        NeuralNetworkClassifier _3HC = new NeuralNetworkClassifier(_3Hidden, increment, train);

        Evaluation _1HE = new Evaluation(train);
        Evaluation _2HE = new Evaluation(train);
        Evaluation _3HE = new Evaluation(train);
        
        System.out.println("Training epochs, 1 hidden layer, 2 hidden layers, 3 hidden layers");
        
        for (int epochs = 0; epochs <= 50000; epochs += increment) {
            _1HE.evaluateModel(_1HC, test);
            _2HE.evaluateModel(_2HC, test);
            _3HE.evaluateModel(_3HC, test);
            
            System.out.print(epochs);
            System.out.print("," + _1HE.pctCorrect());
            System.out.print("," + _2HE.pctCorrect());
            System.out.print("," + _3HE.pctCorrect());
            System.out.println();
            
            _1HC.buildClassifier(train);
            _2HC.buildClassifier(train);
            _3HC.buildClassifier(train);
        }
//        System.out.println("_1HC:\n" + _1HC.toString());//For verification of network structures
//        System.out.println("_2HC:\n" + _2HC.toString());
//        System.out.println("_3HC:\n" + _3HC.toString());
    }
}
