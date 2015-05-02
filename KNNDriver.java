package knnclassifier;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;
import weka.classifiers.lazy.IBk;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ejones23
 */
public class Driver {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        //DataSource source = new DataSource("data/iris.csv");
        DataSource source = new DataSource("data/car.csv");

        Instances data = source.getDataSet();

        int numAttributes = data.numAttributes();
        if (data.classIndex() == -1) {
            data.setClassIndex(numAttributes - 1);
        }

        //Nominalize the data, if needed
        for (int i = 0; i < numAttributes; i++) {
            if (data.attribute(i).isString()) {
                StringToNominal stnFilter = new StringToNominal();
                stnFilter.setAttributeRange((i + 1) + "");
                stnFilter.setInputFormat(data);
                data = Filter.useFilter(data, stnFilter);
            }   
        }

        Standardize standardizeFilter = new Standardize();
        standardizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, standardizeFilter);

        data.randomize(new Random(7));

        //Add a new attribute representing the "distance" to a specific instance
        Add addFilter = new Add();
        addFilter.setAttributeIndex("last");
        addFilter.setAttributeName("distance from new instance");
        addFilter.setInputFormat(data);
        data = Filter.useFilter(data, addFilter);

        String[] options = weka.core.Utils.splitOptions("-P 30");
        RemovePercentage remove = new RemovePercentage();
        remove.setOptions(options);
        remove.setInputFormat(data);
        Instances train = Filter.useFilter(data, remove);

        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        Instances test = Filter.useFilter(data, remove);

        IBk ibk = new IBk();
        ibk.buildClassifier(train);
        Evaluation stdEval = new Evaluation(train);

        MykNNClassifier myKNN = new MykNNClassifier();
        myKNN.buildClassifier(train);
        Evaluation myEval = new Evaluation(train);
            
        int maxNeighbors = 105;
        for (int k = 1; k <= maxNeighbors; k++) {
            ibk.setKNN(k);
            stdEval.evaluateModel(ibk, test);

            myKNN.setKNN(k);
            myEval.evaluateModel(myKNN, test);

            System.out.println(k + "," + stdEval.pctCorrect() + "," 
                    + myEval.pctCorrect());
        }
    }
}
