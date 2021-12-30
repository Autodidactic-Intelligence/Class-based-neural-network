/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package basicneuralnetwork;

import java.util.Random;



/**
 *
 * @author Adam
 */
public class BasicNeuralNetwork {
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        xor();
    }
    
    

    public static void xor(){
        int[] seed = {2,2,1};
        double[] target = {0.0};
        Network network = new Network(seed);
        network.setLearningRate(0.2);
        Random random = new Random();
        int lastLayer = seed[seed.length-1];
        
        double[] ii = {1.0,1.0};
        double[] oo = {0,0};
        double[] io = {1,0};
        double[] oi = {0,1};
        
        for (int i = 0 ; i < 10000; i++){
            target[0]=0.1; //target[1]=1.0;
            network.run(ii);
            network.backpropagation(1, target);
            //network.newBatch();
           
            target[0]=0.1; //target[1]=0;
            network.run(oo);
            network.backpropagation(1, target);
            //network.newBatch();
            
            target[0]=0.9; //target[1]=0;
            network.run(io);            
            network.backpropagation(1, target);
            //network.newBatch();
            
            target[0]=0.9; //target[1]=0;
            network.run(oi);
            network.backpropagation(1,target);
            network.newBatch();
        }
        
        network.run(ii);
        network.printOutput();
        network.run(oo);
        network.printOutput();
        network.run(io);
        network.printOutput();
        network.run(oi);
        network.printOutput();
        network.toString();
    }
}



