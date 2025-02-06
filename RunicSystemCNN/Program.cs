using System;
using System.Collections.Generic;
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

    class Program
    {
        static void Main()
        {
            var mlContext = new MLContext();

            // Create a simple dataset
            Console.WriteLine("Generating dummy data...");
            var dummyData = new List<DummyData>
            {
                new() { Feature1 = 0.2f, Feature2 = 0.5f, Label = 0 },
                new() { Feature1 = 0.8f, Feature2 = 0.3f, Label = 1 },
                new() { Feature1 = 0.6f, Feature2 = 0.7f, Label = 0 },
                new() { Feature1 = 0.1f, Feature2 = 0.9f, Label = 1 },
                new() { Feature1 = 0.4f, Feature2 = 0.2f, Label = 0 }
            };

            // Load dummy data
            Console.WriteLine("Loading dummy data...");
            var dataView = mlContext.Data.LoadFromEnumerable(dummyData);

            // Define the pipeline
            Console.WriteLine("Defining the pipeline...");
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DummyData.Feature1), nameof(DummyData.Feature2))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(dataView);

            Console.WriteLine("Saving the model...");
            mlContext.Model.Save(model, dataView.Schema, "model.zip");

            Console.WriteLine("Loading the model for prediction...");
            var loadedModel = mlContext.Model.Load("model.zip", out _);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<DummyData, DummyPrediction>(loadedModel);

            // Make a prediction
            Console.WriteLine("Making a prediction...");
            var sampleData = new DummyData { Feature1 = 0.7f, Feature2 = 0.4f };
            var prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
        }
    }
}
