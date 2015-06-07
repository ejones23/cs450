package neuralnetworkclassifier;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ejones23
 */
public class StandardComparisonNN {

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
        
        int numNodesHL = 6;
        int epochs = 50000;
        int[] structure = {numNodesHL, data.numClasses()};
        String standardHidden = numNodesHL + "";

        NeuralNetworkClassifier myNN = new NeuralNetworkClassifier(structure, epochs, train);
        MultilayerPerceptron standard = new MultilayerPerceptron();
        standard.setHiddenLayers(standardHidden);
        standard.setTrainingTime(epochs);
        
        myNN.buildClassifier(train);
        standard.buildClassifier(train);

        Evaluation myEval = new Evaluation(train);
        Evaluation stdEval = new Evaluation(train);
        
        myEval.evaluateModel(myNN, test);
        stdEval.evaluateModel(standard, test);
            
        System.out.println(myEval.toSummaryString("\nResults (mine)\n======\n", false));
        System.out.println(stdEval.toSummaryString("\nResults (standard)\n======\n", false));
    }
}
