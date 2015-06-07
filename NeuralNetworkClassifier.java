package neuralnetworkclassifier;

import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author ejones23
 */
public class NeuralNetworkClassifier extends AbstractClassifier {
    private final Network network;
    private int[] inputsPerAttribute;
    private int numInputs;
    private int maxEpochs;
//    private final int[] numNodes;
    
    public NeuralNetworkClassifier(int[] numNodes, int maxEpochs, Instances instances) {
//        this.numNodes = numNodes;
        this.maxEpochs = maxEpochs;
        initializeInputs(instances);
        network = new Network(numNodes, numInputs);
    }
    @Override
    public void buildClassifier(Instances train) throws Exception {
        
        //Separate validation set - TODO
//        String[] options = weka.core.Utils.splitOptions("-P 10");
//        RemovePercentage remove = new RemovePercentage();
//        remove.setOptions(options);
//        remove.setInputFormat(instances);
//        Instances train = Filter.useFilter(instances, remove);
//        remove.setInvertSelection(true);
//        remove.setInputFormat(instances);
//        Instances validation = Filter.useFilter(instances, remove);
        
//        double epochError;
        
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            for (Instance trainInstance : train) {
                double[] inputs = toDoubleArray(trainInstance);
                double[] targets = targets(trainInstance);
                network.feedForward(inputs);
                network.propogateBack(targets);
            }
            
//            epochError = averageError(validation);
//            System.out.println("Validation error: " + epochError + " (epoch " + epoch + ")");
            
            train.randomize(new Random(7));//Sequential (non-batch) learning
        }
    }
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double[] inputs = toDoubleArray(instnc);
        double[] outputs = network.feedForward(inputs);
        return indexOfMax(outputs);
    }
    public void setTrainingTime(int epochs) {
        this.maxEpochs = epochs;
    }
    public String toString() {
        return network.toString();
    }
    private double averageError(Instances validation) {
        double averageError = 0;
        for (Instance validationInstance : validation) {
            double[] inputs = toDoubleArray(validationInstance);
            double[] targets = targets(validationInstance);
            double[] outputs = network.feedForward(inputs);
            averageError += averageDist(outputs, targets);
        }
        averageError /= validation.numInstances();
        return averageError;
    }
    private double averageDist(double[] a, double[] b) {
        double avgDistance = 0;
        for (int i = 0; i < a.length; i++) {
            avgDistance += Math.abs(a[i] - b[i]);
        }
        avgDistance /= a.length;
        return avgDistance;
    }
    private void initializeInputs(Instances instances) {
        int totalInputs = 0;
        int numAttr = instances.numAttributes() - 1;
        this.inputsPerAttribute = new int[numAttr];
        for (int i = 0; i < numAttr; i++) {
            Attribute attribute = instances.attribute(i);
            if (attribute.isNominal()) {
                //TODO: binary nominals really need only one input
                int numValues = instances.numDistinctValues(attribute);
                totalInputs += numValues;
                inputsPerAttribute[i] = numValues;
            }
            else {
                totalInputs++;
                inputsPerAttribute[i] = 1;
            }
        }
        this.numInputs = totalInputs;
    }
    private int indexOfMax(double[] values) {
        int index = 0;
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                index = i;
            }
        }
        return index;
    }
    private double[] toDoubleArray(Instance instance) {
        double[] array = new double[numInputs];
        int numAttr = instance.numAttributes() - 1;
        int base = 0;
        for (int attr = 0; attr < numAttr; attr++) {
            if (instance.attribute(attr).isNominal()) {
                int index = base + (int)instance.value(attr);
                array[index] = 1;
                base += inputsPerAttribute[attr];
            }
            else {
                array[base] = instance.value(base);
                base++;
            }
        }
        return array;
    }
    private double[] targets(Instance instance) {
        double[] targets = new double[instance.numClasses()];
        targets[(int)instance.value(instance.classIndex())] = 1;
        return targets;//All others are initialized to zero
    }
}
