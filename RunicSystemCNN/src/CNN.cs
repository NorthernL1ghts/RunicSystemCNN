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
        private static readonly string ROOT_DIR = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string OUTPUT_DIR = Path.Combine(ROOT_DIR, "output");
        private static readonly string MODEL_PATH = Path.Combine(OUTPUT_DIR, "model.zip");

        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<DummyData, DummyPrediction> _predictionEngine;

        public CNN()
        {
            _mlContext = new MLContext();
            CreateOutputFolder();
            _model = TrainModel();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<DummyData, DummyPrediction>(_model);
        }

        private void CreateOutputFolder()
        {
            if (!Directory.Exists(OUTPUT_DIR))
            {
                Directory.CreateDirectory(OUTPUT_DIR);
                Console.WriteLine($"Created output directory: {OUTPUT_DIR}");
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
            var dataView = _mlContext.Data.LoadFromEnumerable(dummyData);

            Console.WriteLine("Defining the pipeline...");
            var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(DummyData.Feature1), nameof(DummyData.Feature2))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(dataView);

            Console.WriteLine("Saving the model...");
            _mlContext.Model.Save(model, dataView.Schema, MODEL_PATH);

            return model;
        }

        public uint Predict(DummyData sampleData)
        {
            return _predictionEngine.Predict(sampleData).PredictedLabel;
        }
    }
}
