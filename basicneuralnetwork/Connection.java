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
public class Connection {
    private Node input;
    private Node output;
    private double weight;
    private double weightChanges;
    private int counter;

    public Connection(Node input, Node output, double weight) {
        this.input = input;
        this.output = output;
        this.weight = weight;
        this.weightChanges = 0;
    }
    
    public double pullThrough() {
        //this.input.resetValue(); //resets value for the next go around
        return this.input.getOutput() * this.weight;
    }
    
    public void addWeightChanges(double weightChange) {
        this.weightChanges += weightChange;
        this.counter++;
    }
    
    public void calculateNewWeight(){
        this.weightChanges /= this.counter;
        this.weight -= this.weightChanges;
        this.weightChanges = 0;
        this.counter = 0;
    }

    public Node getInput() {
        return input;
    }

    public void setInput(Node input) {
        this.input = input;
    }

    public Node getOutput() {
        return output;
    }

    public void setOutput(Node output) {
        this.output = output;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
    
    @Override
    public String toString(){
        String s = "In node| " + this.input.toString() + " | Out node| " +
                this.output.toString() + " | Weight: " + this.weight;
        return s;
    }
    
}
