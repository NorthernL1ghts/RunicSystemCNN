using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace RunicSystemCNN
{
    public class DummyData
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public uint Label { get; set; }
    }

    public class DummyPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel { get; set; }
    }

    public class CNN
    {
        private static readonly string RootDir = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string OutputDir = Path.Combine(RootDir, "output");
        private static readonly string ModelPath = Path.Combine(OutputDir, "model.zip");

        private readonly MLContext mlContext;
        private readonly ITransformer model;
        private readonly PredictionEngine<DummyData, DummyPrediction> predictionEngine;

        public CNN()
        {
            mlContext = new MLContext();
            CreateOutputFolder();
            model = TrainModel();
            predictionEngine = mlContext.Model.CreatePredictionEngine<DummyData, DummyPrediction>(model);
        }

        private void CreateOutputFolder()
        {
            if (Directory.Exists(OutputDir))
            {
                Console.WriteLine($"Output directory found at {OutputDir}, skipping creation.\n");
            }
            else
            {
                Directory.CreateDirectory(OutputDir);
                Console.WriteLine($"Created output directory: {OutputDir}");
            }
        }


        private ITransformer TrainModel()
        {
            Console.WriteLine("Generating dummy data...");
            var dummyData = new List<DummyData>
            {
                new() { Feature1 = 0.2f, Feature2 = 0.5f, Label = 0 },
                new() { Feature1 = 0.8f, Feature2 = 0.3f, Label = 1 },
                new() { Feature1 = 0.6f, Feature2 = 0.7f, Label = 0 },
                new() { Feature1 = 0.1f, Feature2 = 0.9f, Label = 1 },
                new() { Feature1 = 0.4f, Feature2 = 0.2f, Label = 0 }
            };

            Console.WriteLine("Loading dummy data...");
            var dataView = mlContext.Data.LoadFromEnumerable(dummyData);

            Console.WriteLine("Defining the pipeline...");

            // Step 1: Concatenate feature columns
            var featureColumn = mlContext.Transforms.Concatenate("Features", nameof(DummyData.Feature1), nameof(DummyData.Feature2));

            // Step 2: Map label to key
            var labelColumn = mlContext.Transforms.Conversion.MapValueToKey("Label");

            // Step 3: Apply the multiclass classification trainer
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "Features");

            // Step 4: Convert predicted label from key back to value
            var predictedLabelColumn = mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");

            // Combine all transformations and training into a pipeline
            var pipeline = featureColumn.Append(labelColumn)
                                       .Append(trainer)
                                       .Append(predictedLabelColumn);

            Console.WriteLine("Training the model...");
            var trainedModel = pipeline.Fit(dataView);

            Console.WriteLine("Saving the model...");
            mlContext.Model.Save(trainedModel, dataView.Schema, ModelPath);

            return trainedModel;
        }


        public uint Predict(DummyData sampleData)
        {
            return predictionEngine.Predict(sampleData).PredictedLabel;
        }
    }
}
