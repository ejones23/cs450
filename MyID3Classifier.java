package id3classifier;

import java.util.Arrays;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ejones23
 */
public class MyID3Classifier extends AbstractClassifier {
    
    private Instances instances;
    private Node root;
    private static final double LN_2 = Math.log(2);
    private static final String indent = "   ";
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.instances = data;
        boolean[] attrAvail = new boolean[data.numAttributes() - 1];
        Arrays.fill(attrAvail, Boolean.TRUE);
        root = nextFeature(data, attrAvail);
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return classify(root, instance);
    }
    private double classify(Node node, Instance instance) {
        Node[] edges = node.getChildren();
        if (edges == null) {//Is leaf node (label)
            return node.getAttributeValue();
        }
        else {//Is internal node (feature)
            int attributeNum = node.getAttributeValue();
            int edgeNum;
            if (instance.isMissing(attributeNum)) {
                edgeNum = (int)node.getInstances().meanOrMode(attributeNum);
            }
            else {
                edgeNum = (int)instance.value(attributeNum);
            }
            return classify(edges[edgeNum], instance);
        }
    }
    private Node nextFeature(Instances set, boolean[] attrAvail) {
        int[] labelCounts = countLabels(set);
        int mostFrequentLabel = maximum(labelCounts);
        if (labelCounts[mostFrequentLabel] == set.numInstances()) {
            //If all instances have the same label, return that label
            return new Node(set, mostFrequentLabel, null);
        }
        else if (noneRemaining(attrAvail)) {
            //If there are no features left to test, return most frequent label
            return new Node(set, mostFrequentLabel, null);
        }
        else {
            int feature = maxInformationGain(set, attrAvail);
            Node[] branch = new Node[set.attribute(feature).numValues()];
            Instances[] splits = splitByValue(set, feature);
            for (int i = 0; i < branch.length; i++) {
                if (splits[i].numInstances() > 0) {
                    branch[i] = nextFeature(splits[i], newAvail(attrAvail, feature));
                }
                else
                {
                    int guess = (int)set.meanOrMode(set.classIndex());
                    branch[i] = new Node(splits[i], guess, null);
                }
            }
            return new Node(set, feature, branch);
        }
    }
    private int maxInformationGain(Instances set, boolean[] attrAvail) {
        int lowestEntropyFeature = -1;//None found yet
        double lowestEntropy = Double.MAX_VALUE;
        double entropy;
        for (int i = 0; i < attrAvail.length; i++) {
            if (attrAvail[i]) {
                entropy = calcEntropyByFeature(set, i);
                if (entropy < lowestEntropy) {
                    lowestEntropy = entropy;
                    lowestEntropyFeature = i;
                }
            }
        }
        return lowestEntropyFeature;
    }
    private double calcEntropyByFeature(Instances pool, int feature) {
        Instances[] portion = splitByValue(pool, feature);
        double[] entropies = new double[pool.numDistinctValues(feature)];
        double[] weights = new double[entropies.length];
        for (int i = 0; i < portion.length; i++) {
            entropies[i] = calcEntropyOfSet(portion[i]);
            weights[i] = portion[i].numInstances()/(double)pool.numInstances();
        }
        //Calculate the weighted average of the entropies
        double weightedAverage = 0;
        for (int i = 0; i < entropies.length; i++) {
            weightedAverage += entropies[i] * weights[i];
        }
        return weightedAverage;
    }
    private double calcEntropyOfSet(Instances set) {
        int numInst = set.numInstances();
        if (numInst == 0) return 0;
        double p;
        double entropy = 0;
        int[] labelCounts = countLabels(set);
        for (int i = 0; i < labelCounts.length; i++) {
            p = labelCounts[i]/(double)numInst;
            entropy -= plog2p(p);
        }
        return entropy;
    }
    private double plog2p(double p) {
        if (p == 0) {
            return 0;
        }
        else {
            return p * Math.log(p) / LN_2;//p*log2(p)
        }
    }
    private boolean[] newAvail(boolean[] attrAvail, int feature) {
        boolean[] newArray = new boolean[attrAvail.length];
        for (int i = 0; i < attrAvail.length; i++) {
            if (i == feature) {
                newArray[i] = false;
            }
            else {
                newArray[i] = attrAvail[i];
            }
        }
        return newArray;
    }
    private boolean noneRemaining(boolean[] available) {
        for (int i = 0; i < available.length; i++) {
            if (available[i]) {
                return false;
            }
        }
        return true;
    }
    private int maximum(int[] array) {
        int index = -1;
        int maxVal = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                index = i;
            }
        }
        return index;
    }
    private int[] countLabels(Instances set) {
        int[] counts = new int[set.numClasses()];
        int numInst = set.numInstances();
        for (int i = 0; i < numInst; i++) {
            counts[(int)set.get(i).classValue()]++;
        }
        return counts;
    }
    private Instances[] splitByValue(Instances pool, int attr) {
        Instances[] portion = new Instances[pool.numDistinctValues(attr)];
        for (int i = 0; i < portion.length; i++) {
            portion[i] = new Instances(pool, 0, 0);//Create empty Instances
        }
        int numInst = pool.numInstances();
        for (int i = 0; i < numInst; i++) {
            portion[(int)pool.get(i).value(attr)].add(pool.get(i));
        }
        return portion;
    }
    @Override
    public String toString() {
        return toString(root, 0);
    }
    private String toString(Node node, int level) {
        String out = "";
        if (node == null) {
            return "null\n";
        }
        int attributeValue = node.getAttributeValue();
        Attribute attribute = instances.attribute(attributeValue);
        Node[] edges = node.getChildren();
        if (edges == null) {//Is leaf node (label)
            out += instances.classAttribute().value(attributeValue) + "\n";
        }
        else {//Is internal node (feature)
            out += attribute.name() + "?\n";
            for (int i = 0; i < edges.length; i++) {
                for (int j = 0; j <= level; j++) {
                    out += indent;
                }
                out += attribute.value(i) + "-->" + toString(edges[i], level + 1);
            }
        }
        return out;
    }
}
