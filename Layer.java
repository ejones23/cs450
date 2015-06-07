package neuralnetworkclassifier;

/**
 *
 * @author ejones23
 */
public class Layer {
    private final Node[] nodes;
    
    public Layer(int numInputs, int numNodes) {
        nodes = new Node[numNodes];
        for (int i = 0; i < numNodes; i++) {
            nodes[i] = new Node(numInputs);
        }
    }
    public double[] outputs(double[] inputs) {
        double[] outputs = new double[nodes.length];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = nodes[i].output(inputs);
        }
        return outputs;
    }
    public Node[] getNodes() {
        return nodes;
    }
    public double[] errors(double[] expected, Layer outerLayer) {
        double[] errors = new double[nodes.length];
        for (int i = 0; i < errors.length; i++) {
            Node node = nodes[i];
            double produced = node.getLastOutput();
            if (outerLayer == null) {//Is output layer
                errors[i] = (produced - expected[i]) * produced * (1 - produced);
                node.setError(errors[i]);
            }
            else {//Is hidden layer
                errors[i] = produced * (1 - produced) * outerLayer.weightedSumOfErrors(i);
                node.setError(errors[i]);
            }
        }
        return errors;
    }
    public double weightedSumOfErrors(int inputNode) {
        double sum = 0;
        for (int i = 0; i < nodes.length; i++) {
            Node node = nodes[i];
            double weight = node.getWeights()[inputNode + 1];//The bias node makes this a count-from-1 array
            sum += weight * node.getError();
        }
        return sum;
    }
    //IMPORTANT: should be called after a call to errors (TODO: enforce this with class structure)
    public void updateWeights() {
        for (Node node : nodes) {
            node.updateWeights();
        }
    }
}
