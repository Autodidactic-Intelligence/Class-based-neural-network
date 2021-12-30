/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package basicneuralnetwork;
/**
 *
 * @author Adam
 */
public class Node {
    private final int layer; //layer the node is in
    private double value; //pre activation and bias value
    private double bias; //bias
    private double output; //final output, bias and activation function used
    private double netToOut;
    private double derivative; //stores the product of the derivatives of the nodes before it
    private Connection[] backwardConns; //forward connections
    

    public Node(int layer, double value, double bias) {
        this.layer = layer;
        this.value = value;
        this.bias = bias;
        this.backwardConns = null;
        this.derivative = 0;
    }
    
    //does the actual backpropagation logic for the node
    public void backPropagate(double learningRate){
        for (Connection c: backwardConns){
            double weightChanges = learningRate * this.derivative * c.getInput().getOutput() * this.netToOut;
            c.addWeightChanges(weightChanges);
            
            this.bias -= this.derivative * this.netToOut * learningRate;
            
            double error = this.derivative * c.getWeight();
            c.getInput().addDerivative(error);
        }
    }
    
    //Calculates the final output from the value and bias
    public void calculateOutput(){
        double funcIn = this.value + this.bias;
        double denominator = 1 + Math.exp(-funcIn);
        this.output = 1/denominator;
        //derivative of sigmoid function
        this.netToOut = this.output * (1-this.output);
        this.value = 0;
    }
    
    //pulls through values from the previous layer
    public void getInputs(){
        for (Connection c: backwardConns) this.value += c.pullThrough();
    }
    
    //Sets up forward facing connections
    public void setBackwardConns(Connection[] conns){
        this.backwardConns = new Connection[conns.length];
        System.arraycopy(conns, 0, backwardConns, 0, conns.length);
    }

    public double getDerivative() {
        return derivative;
    }
    
    public void addDerivative(double derivative){
        this.derivative += derivative;
    }

    public void setDerivative(double derivative) {
        this.derivative = derivative;
    }
    
    public double getOutput(){
        return output;
    }
    
    //Used only for the input nodes
    public void setOutput(double x){
        this.output = x;
    }

    public int getLayer() {
        return layer;
    }

    public double getValue() {
        return value;
    }
    //value is always added to, never explicitly set.
    public void setValue(double value) {
        this.value = value;
    }
    //Used to reset all values to zero
    public void resetValue() {
        this.value = 0;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public Connection[] getBackwardConns() {
        return backwardConns;
    }
    
    public double getNetToOut() {
        return netToOut;
    }

    public void setNetToOut(double netToOut) {
        this.netToOut = netToOut;
    }
    
    @Override
    public String toString(){
        String s = "Layer: " + this.layer +" output: " + this.output + " Bias: " + this.bias;
        return s;
    }
    
    public void printConnections(){
        for (Connection c: this.backwardConns){
            System.out.println(c.toString());
        }
    } 
}
