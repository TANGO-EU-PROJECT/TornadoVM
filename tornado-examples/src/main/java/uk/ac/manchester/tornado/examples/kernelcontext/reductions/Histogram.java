/*
 * Copyright (c) 2025, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.examples.kernelcontext.reductions;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.List;
import java.util.Random;

/**
 * <p>
 * How to run?
 * </p>
 * <code>
 * $ tornado --threadInfo -m tornado.examples/uk.ac.manchester.tornado.examples.kernelcontext.reductions.Histogram 1024 256
 * </code>
 */
public class Histogram {

    private static int numBins = 4;
    private static int blockSize = 256;
    private static int size = 256;

    private static IntArray dataPoints;
    private static IntArray histDataTornado;
    private static IntArray histDataJava;

    private static long javaStartTime;
    private static long javaEndTime;
    private static long tornadoStartTime;
    private static long tornadoEndTime;

    public Histogram(int size, int numberOfBins) {
        int[] inputData = createDataPoints(size, numberOfBins);
        setInputs(inputData, numberOfBins);
    }

    public static int[] createDataPoints(int numDataPoints, int numberOfBins) {
        Random rand = new Random();
        int[] inputData = new int[numDataPoints];
        // Initialize input data with random numbers
        for (int i = 0; i < numDataPoints; i++) {
            inputData[i] = rand.nextInt(numberOfBins);
        }
        return inputData;
    }

    /**
     * This method implements the following CUDA kernel with the TornadoVM Kernel API.
     *
     * __global__ void histogramKernel(int *data, int *hist, int dataSize) {
     * int tid = threadIdx.x + blockIdx.x * blockDim.x;
     *
     * if (tid < dataSize) {
     * atomicAdd(&hist[data[tid]], 1);
     * }
     * }
     *
     * @param context
     * @param input
     * @param output
     */
    public static void histogramKernel(KernelContext context, IntArray input, IntArray output) {
        int tid = context.globalIdx;

        if (tid < input.getSize()) {
            int index = input.get(tid);
            context.atomicAdd(output, index, 1);
        }
    }

    public static void histogram(KernelContext context, IntArray input, IntArray output) {
        for (int tid = 0; tid < input.getSize(); tid++) {
            int index = input.get(tid);
            context.atomicAdd(output, index, 1);
            output.set(index, output.get(index));
        }
    }

    public static void setInputs(int[] inputData, int numberOfBins) {
        size = inputData.length;
        dataPoints = IntArray.fromArray(inputData);
        histDataTornado = new IntArray(inputData.length);
        histDataJava = new IntArray(inputData.length);
        numBins = numberOfBins;
    }

    public static void setBlockSize(int blockSize) {
        Histogram.blockSize = blockSize;
    }

    public static IntArray runWithGPU() {
        // 1. Create a TaskGraph for the assign cluster method
        KernelContext context = new KernelContext();
        WorkerGrid workerGrid = new WorkerGrid1D(size);
        workerGrid.setGlobalWork(size, 1, 1);
        workerGrid.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", workerGrid);

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, dataPoints) //
                .task("t0", Histogram::histogramKernel, context, dataPoints, histDataTornado) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, histDataTornado); //

        // 2. Create an execution plan
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());

        // 3. Execute the plan - histogram with TornadoVM
        tornadoStartTime = System.nanoTime();
        try (executionPlan) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        tornadoEndTime = System.nanoTime();

        return histDataTornado;
    }

    public static IntArray runWithJava() {
        // Run histogram in Java
        KernelContext context = new KernelContext();

        javaStartTime = System.nanoTime();
        histogram(context, dataPoints, histDataJava);
        javaEndTime = System.nanoTime();

        return histDataJava;
    }

    public static void reportTornadoExecutionTime() {
        System.out.println("Total time of Java execution: " + (tornadoEndTime - tornadoStartTime) + " (nanoseconds)");
    }

    public static void reportJavaExecutionTime() {
        System.out.println("Total time of Java execution: " + (javaEndTime - javaStartTime) + " (nanoseconds)");
    }

    public static int[] fromList(List<Integer> list) {
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (args.length == 1) {
            size = Integer.parseInt(args[0]);
        } else if (args.length == 2) {
            size = Integer.parseInt(args[0]);
            setBlockSize(Integer.parseInt(args[1]));
        }

        Histogram histogram = new Histogram(size, numBins);
        IntArray javaHistData = histogram.runWithJava();
        IntArray tornadoHistData = histogram.runWithGPU();

        final boolean valid = validate(tornadoHistData, javaHistData);

        if (!valid) {
            System.out.println(" ................ [FAIL]");
        } else {
            System.out.println(" ................ [PASS]");
        }
    }

    private static boolean validate(IntArray histDataTornado, IntArray histDataJava) {
        int counter = 0;
        for (int i = 0; i < numBins + 1; i++) {
            counter += histDataTornado.get(i);
            if (histDataJava.get(i) != histDataTornado.get(i)) {
                System.out.println("[FAIL] histDataJava.get(" + i + "): " + histDataJava.get(i) + " - histDataTornado.get(" + i + "): " + histDataTornado.get(i));
                return false;
            }
        }
        return counter == histDataTornado.getSize();
    }
}
