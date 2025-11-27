using System;
using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    private MLPParameters mlpParameters;
    private List<float[,]> _coefs;
    private List<float[]> _intercepts;

    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
        _coefs = mlpParameters.GetCoeff();
        _intercepts = mlpParameters.GetInter();
    }

    /// <summary>
    /// Ejecuta el feedforward de la red.
    /// El vector input YA debe venir con el mismo preprocesado que en Python
    /// (mismas columnas, mismo orden, mismo escalado, mismo OHE).
    /// </summary>
    public float[] FeedForward(float[] input)
    {
        float[] a = input; // activaciones iniciales (capa de entrada)

        for (int layer = 0; layer < _coefs.Count; layer++)
        {
            float[,] W = _coefs[layer];
            float[] b = _intercepts[layer];

            int inDim = W.GetLength(0);
            int outDim = W.GetLength(1);

            if (a.Length != inDim)
            {
                Debug.LogError($"Dimensión de entrada incorrecta en capa {layer}. Esperado {inDim}, recibido {a.Length}");
                return new float[outDim];
            }

            // z = W^T * a + b   (en sklearn: shape (n_in, n_out))
            float[] z = new float[outDim];
            for (int j = 0; j < outDim; j++)
            {
                float sum = 0f;
                for (int i = 0; i < inDim; i++)
                    sum += a[i] * W[i, j];

                sum += b[j];
                z[j] = sum;
            }

            // Si no es la última capa -> sigmoide
            if (layer < _coefs.Count - 1)
            {
                float[] nextA = new float[outDim];
                for (int j = 0; j < outDim; j++)
                    nextA[j] = sigmoid(z[j]);
                a = nextA;
            }
            else
            {
                // Última capa -> softmax
                a = SoftMax(z);
            }
        }

        return a;
    }

    /// <summary>
    /// Cálculo de la sigmoidal
    /// </summary>
    private float sigmoid(float z)
    {
        // usar Mathf.Exp para no mezclar double/float
        return 1.0f / (1.0f + Mathf.Exp(-z));
    }

    /// <summary>
    /// Softmax sobre el vector de la última capa.
    /// </summary>
    public float[] SoftMax(float[] zArr)
    {
        int n = zArr.Length;
        float[] result = new float[n];

        // estabilización numérica: restar el máximo
        float max = zArr[0];
        for (int i = 1; i < n; i++)
            if (zArr[i] > max) max = zArr[i];

        float sumExp = 0f;
        for (int i = 0; i < n; i++)
        {
            float e = Mathf.Exp(zArr[i] - max);
            result[i] = e;
            sumExp += e;
        }

        if (sumExp == 0f) // por si acaso
            return result;

        for (int i = 0; i < n; i++)
            result[i] /= sumExp;

        return result;
    }

    /// <summary>
    /// Elige el índice de mayor valor en el vector de salida.
    /// </summary>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el índice del mayor valor y devuelve ese valor en max.
    /// </summary>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        int index = 0;
        max = output[0];

        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }

        return index;
    }
}
