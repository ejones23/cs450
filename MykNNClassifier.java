package knnclassifier;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ejones23
 */
public class MykNNClassifier extends AbstractClassifier {

    private Instances instances;
    private int k;//number of nearest neighbors
    @Override
    public void buildClassifier(Instances i) throws Exception {
        this.instances = i;
    }
    public void setKNN(int k) {
        this.k = k;
    }
    @Override
    public double classifyInstance(Instance pInst) throws Exception {
        int distIndex = pInst.classIndex() + 1;
        for (Instance mInst : instances) {
            mInst.setValue(distIndex, distance(mInst, pInst));
        }
        instances.sort(distIndex);//The first k are the "k nearest neighbors"
        Instances kNN = new Instances(instances, 0, this.k);
        return kNN.meanOrMode(kNN.classIndex());
    }
    private double distance(Instance inst1, Instance inst2) {
        double squares = 0;
        int classIndex = instances.classIndex();
        for (int i = 0; i < classIndex; i++) {
            squares += Math.pow(inst1.value(i) - inst2.value(i), 2);
        }
        return squares;//No need to take the root, since we're only interested in the relative distances
    }
}
