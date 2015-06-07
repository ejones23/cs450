package neuralnetworkclassifier;

import java.util.Random;

/**
 *
 * @author ejones23
 */
public class Node {
    private final double[] weights;
    private double lastOutput;
    private double[] lastInputs;
    private double error;
    private static final double learningRate = 0.3;//Best between 0.1 and 0.4
    private static final double beta = 0.2;//Best below 0.3
    private static final Random rand = new Random(7);//Arbitrary seed
    
    public Node(int numInputs) {
        weights = new double[numInputs + 1];//Adding a weight for the bias input
        double denom = Math.sqrt(numInputs) / 2.0;
        for (int i = 0; i <= numInputs; i++) {
            weights[i] = (rand.nextDouble() - 0.5) / denom;
        }
    }
    public double output(double[] inputs) {
        lastInputs = inputs;
        lastOutput = activation(weightedSum(inputs));
        return lastOutput;
    }
    public void setError(double error) {
        this.error = error;
    }
    public double getError() {
        return error;
    }
    public double getLastOutput() {
        return lastOutput;
    }
    public double[] getLastInputs() {
        return lastInputs;
    }
    public double[] getWeights() {
        return weights;
    }
    private double weightedSum(double[] inputs) {
        double sum = -1.0 * weights[0];//The bias input
        for (int i = 1; i < weights.length; i++) {
            sum += inputs[i - 1] * weights[i];
        }
        return sum;
    }
    private double activation(double sum) {
        return 1 / (1 + Math.exp(-beta * sum));//Sigmoid function
    }
    public void updateWeights() {
        double LRxERR = learningRate * error;
        weights[0] = weights[0] - LRxERR * (-1.0);//Bias input
        for (int i = 0; i < lastInputs.length; i++) {
            weights[i + 1] = weights[i + 1] - LRxERR * lastInputs[i];
        }
    }
}
