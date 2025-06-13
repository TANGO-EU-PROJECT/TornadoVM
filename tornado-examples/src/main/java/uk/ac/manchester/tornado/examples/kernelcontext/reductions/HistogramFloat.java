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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.List;
import java.util.Random;

/**
 * <p>
 * How to run?
 * </p>
 * <code>
 * $ tornado --threadInfo -m tornado.examples/uk.ac.manchester.tornado.examples.kernelcontext.reductions.HistogramFloat 1024 256
 * </code>
 */
public class HistogramFloat {

    private static int numBins = 4;
    private static int blockSize = 256;
    private static int size = 256;

    private static FloatArray dataPoints;
    private static FloatArray histDataTornado;
    private static FloatArray histDataJava;

    public HistogramFloat(int size, int numberOfBins) {
        float[] inputData = createDataPoints(size, numberOfBins);
        setInputs(inputData, numberOfBins);
    }

    public static float[] createDataPoints(int numDataPoints, int numberOfBins) {
        Random rand = new Random();
        float[] inputData = new float[numDataPoints];
        // Initialize input data with random numbers
        for (int i = 0; i < numDataPoints; i++) {
            inputData[i] = rand.nextFloat(numberOfBins);
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
    public static void histogramKernel(KernelContext context, FloatArray input, FloatArray output) {
        int tid = context.globalIdx;

        if (tid < input.getSize()) {
            int index = (int) input.get(tid);
            context.atomicAdd(output, index, 1);
        }
    }

    public static void histogram(KernelContext context, FloatArray input, FloatArray output) {
        for (int tid = 0; tid < input.getSize(); tid++) {
            int index = (int) input.get(tid);
            context.atomicAdd(output, index, 1);
            output.set(index, output.get(index));
        }
    }

    public static void setInputs(float[] inputData, int numberOfBins) {
        size = inputData.length;
        dataPoints = FloatArray.fromArray(inputData);
        histDataTornado = new FloatArray(inputData.length);
        histDataJava = new FloatArray(inputData.length);
        numBins = numberOfBins;
    }

    public static void setBlockSize(int blockSize) {
        HistogramFloat.blockSize = blockSize;
    }

    public static FloatArray runWithGPU() {
        // 1. Create a TaskGraph for the assign cluster method
        KernelContext context = new KernelContext();
        WorkerGrid workerGrid = new WorkerGrid1D(size);
        workerGrid.setGlobalWork(size, 1, 1);
        workerGrid.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", workerGrid);

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, dataPoints) //
                .task("t0", HistogramFloat::histogramKernel, context, dataPoints, histDataTornado) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, histDataTornado); //

        // 2. Create an execution plan
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());

        // 3. Execute the plan - histogram with TornadoVM
        long start = System.nanoTime();
        try (executionPlan) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        long end = System.nanoTime();
        System.out.println("Total time of TornadoVM execution: " + (end - start) + " (nanoseconds)");

        return histDataTornado;
    }

    public static FloatArray runWithJava() {
        // Run histogram in Java
        KernelContext context = new KernelContext();

        long start = System.nanoTime();
        histogram(context, dataPoints, histDataJava);
        long end = System.nanoTime();

        System.out.println("Total time of Java execution: " + (end - start) + " (nanoseconds)");

        return histDataJava;
    }

    public static float[] fromList(List<Float> list) {
        float[] result = new float[list.size()];
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
