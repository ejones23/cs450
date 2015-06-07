package neuralnetworkclassifier;

/**
 *
 * @author ejones23
 */
public class Network {
    private final Layer[] layers;
    
    public Network(int[] numNodes, int numFirstInputs) {
        layers = new Layer[numNodes.length];
        layers[0] = new Layer(numFirstInputs, numNodes[0]);
        for (int i = 1; i < numNodes.length; i++) {
            layers[i] = new Layer(numNodes[i - 1], numNodes[i]);
        }
    }
    public double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.outputs(inputs);
        }
        return inputs;
    }
    public void propogateBack(double[] targets) {
        setErrors(targets);
        updateWeights();
    }
    public Layer[] getLayers() {
        return layers;
    }
    public String toString() {
        String table = "";
        for (int i = 0; i < layers.length; i++) {
            Node[] nodes = layers[i].getNodes();
            table += "Layer " + i + ":\n";
            for (int j = 0; j < nodes.length; j++) {
                double[] weights = nodes[j].getWeights();
                table += "  Node " + j + ": ";
                for (int k = 0; k < weights.length; k++) {
                    table += " " + weights[k];
                }
                table += "\n";
            }
        }
        return table;
    }
    private void setErrors(double[] targets) {
        //First, the output layer
        int indexOfLastLayer = layers.length - 1;
        Layer outputLayer = layers[indexOfLastLayer];
        Node[] outputNodes = outputLayer.getNodes();
        for (int i = 0; i < outputNodes.length; i++) {
            Node node = outputNodes[i];
            double output = node.getLastOutput();
            double error = output * (1 - output) * (output - targets[i]);
            node.setError(error);
        }
        //Then the hidden layer(s)
        for (int iLayer = indexOfLastLayer - 1; iLayer >= 0; iLayer--) {
            Layer leftLayer = layers[iLayer];
            Layer rightLayer = layers[iLayer + 1];
            Node[] leftNodes = leftLayer.getNodes();
            Node[] rightNodes = rightLayer.getNodes();
            for (int iNode = 0; iNode < leftNodes.length; iNode++) {
                Node leftNode = leftNodes[iNode];
                double output = leftNode.getLastOutput();
                double weightedSumErrors = weightedSumErrors(iNode, rightNodes);
                double error = output * (1 - output) * weightedSumErrors;
                leftNode.setError(error);
            }
        }
    }
    private double weightedSumErrors(int iInput, Node[] nodes) {
        double sum = 0;
        for (Node node : nodes) {
            double weight = node.getWeights()[iInput + 1];//+1 for bias inputs
            sum += weight * node.getError();
        }
        return sum;
    }
    private void updateWeights() {
        //IMPORTANT: All errors should be updated before updating weights
        for (Layer layer : layers) {
            layer.updateWeights();
        }
    }
}
