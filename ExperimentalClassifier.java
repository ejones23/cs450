/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package experimentalclassifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Random;
import java.util.Vector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ejones23
 */
public class ExperimentalClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        DataSource source = new DataSource("data/iris.csv");
        
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        data.randomize(new Random());
        
        String[] options = weka.core.Utils.splitOptions("-P 30");
        RemovePercentage remove = new RemovePercentage();
        remove.setOptions(options);
        remove.setInputFormat(data);
        Instances train = Filter.useFilter(data, remove);
        
        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        Instances test = Filter.useFilter(data, remove);
        
        Classifier classifier = new HardCodedClassifier();
        classifier.buildClassifier(train);//Currently, this does nothing
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(classifier, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}
