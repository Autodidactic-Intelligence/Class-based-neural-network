/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package basicneuralnetwork;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;
/**
 *
 * @author Adam
 */
public class Network {
    private double learningRate; //learning rate of the network
    
    private int[] seed; //the array used to create the network
    private double[] cost;
    //JAGGED 2d array of all nodes; 2d sectioning shows layers
    private Node[][] nodeList;
    //List of all connections
    private ArrayList<Connection> connectionList;
    
    
    //Takes an array of ints to generate the network
    //Length of the array is the amount of layers; 
    //The integers in each position represent how many nodes in that layer
    //Example: input array of {4,3,2}: 
    //3 layers: 4 nodes in the 1st layer, 3 in the 2nd, and 2 in the 3rd
    public Network(int[] layers){
        this.learningRate = 0.5;
        
        //stores the array used to create the network
        this.seed = new int[layers.length];
        System.arraycopy(layers, 0, this.seed, 0, layers.length);
        
        //cost vector for the network
        cost = new double[this.seed[this.seed.length-1]];
        
        connectionList = new ArrayList<>();
        
        final int ZERO = 0; //Starting value of a node
        final int BIAS = 1; //Starting bias of a node
        Random random = new Random(); //used for random starting weights
        
        //Sets up the nodeList array with the proper amount spots
        nodeList = new Node[layers.length][];
        for (int i=0; i<layers.length; i++) nodeList[i] = new Node[layers[i]];
        
        //Generates the nodes
        for (int j = 0; j < layers.length; j++){
            for (int i = 0; i < layers[j]; i++){
                Node node = new Node(j, ZERO, BIAS);
                nodeList[j][i] = node;
            }
        }
        //Generates the connections and sets them for each node
        for (int j = 1; j < nodeList.length; j++){
            for (Node en: nodeList[j]) {
                Connection[] conns = new Connection[nodeList[j-1].length];
                for (int i = 0; i < nodeList[j-1].length; i++) {
                    Connection conn = new Connection(nodeList[j-1][i], en, random.nextDouble());
                    conns[i] = conn;
                    connectionList.add(conn);
                }
                en.setBackwardConns(conns);
            }
        }
    }
    
    public double[] run(double[] inputs){
        double[] results = new double[seed[seed.length-1]];
        //ensures the number of inputs matches the number of input nodes
        if (inputs.length != nodeList[0].length) return null;
        
        //sets the output value of the input nodes to their respective values
        for (int i = 0; i < nodeList[0].length; i++){
            nodeList[0][i].setOutput(inputs[i]);
        }
        
        //Runs through the reamining nodes
        //If logic used to determine if the last layer has been reached
        for (int i = 1; i < nodeList.length; i++){
            for (Node n : nodeList[i]) {n.getInputs(); n.calculateOutput();}
            if (i == nodeList.length-1){
                for (int k = 0; k < nodeList[i].length; k++){
                    results[k] = nodeList[i][k].getOutput();
                }
            }
        }
        return results;
    }
    
    public void backpropagation(int trainingNum, double[] wanted){
        int lastLayer = this.seed.length-1;
        
         for (int i = 0; i < this.seed[lastLayer]; i++){
//             if (wanted[i] == 1){
//                 this.cost[i] = -Math.log(nodeList[lastLayer][i].getOutput());
//             }
//             else{
//                 this.cost[i] = -Math.log(1-nodeList[lastLayer][i].getOutput());
//             }
            this.cost[i] = -(wanted[i] - nodeList[lastLayer][i].getOutput());
        }
        for (double d : cost) d /= trainingNum;

        for (int i = 0; i < nodeList[lastLayer].length; i++){
            nodeList[lastLayer][i].setDerivative(this.cost[i]);
        }
        
        for (int i = lastLayer; i > 0; i--){
            for(Node node : nodeList[i]){
                node.backPropagate(this.learningRate); 
            }
        }
        for (int i = this.seed.length-1; i > 0; i--){
            for(Node node : nodeList[i]){
                node.setDerivative(0);
            }
        }
    }
    
    public void resetDerivatives(){
        for (int i =0; i < nodeList.length; i++){
            for (Node n : nodeList[i]) n.setDerivative(0);
        }
    }
    
    //Readies the network for a new batch
    public void newBatch(){
        for (Connection c : this.connectionList){
            c.calculateNewWeight();
        }
        for (double d: cost) d = 0;
        for (int i = 1; i < this.seed.length; i++){
            for (Node n: nodeList[i]) n.setDerivative(0);
        }
    }
     
    public void setLearningRate(double rate){
        this.learningRate = rate;
    }

    public double getLearningRate() {
        return learningRate;
    }
    
    public void printOutput(){
        for (Node n : nodeList[seed.length-1]) System.out.println(n.getOutput());
        System.out.println();
    }
    
    public void printBiases(){
        for (Node n : nodeList[seed.length-1]) System.out.println(n.getBias());
        System.out.println();
    }
    
    @Override
    public String toString(){
        for (int i = 0; i < nodeList.length; i++){
            for (Node n: nodeList[i]){
                if (n.getBackwardConns() != null) n.printConnections();
            }
        }
        System.out.println();
        return null;
    }
}
