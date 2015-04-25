/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package experimentalclassifier;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ejones23
 */
public class HardCodedClassifier extends AbstractClassifier {

    @Override
    public void buildClassifier(Instances i) throws Exception {
        //Empty
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        return 0;//Hard-coded to simply return the first class
    }
}
