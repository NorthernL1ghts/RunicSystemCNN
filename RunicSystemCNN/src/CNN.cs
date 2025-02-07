using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace RunicSystemCNN.src
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
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<DummyData, DummyPrediction> _predictionEngine;

        public CNN()
        {
            _mlContext = new MLContext();
            _model = TrainModel();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<DummyData, DummyPrediction>(_model);
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
            _mlContext.Model.Save(model, dataView.Schema, "model.zip");

            return model;
        }

        public uint Predict(DummyData sampleData)
        {
            return _predictionEngine.Predict(sampleData).PredictedLabel;
        }
    }
}
