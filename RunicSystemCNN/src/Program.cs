using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RunicSystemCNN.src
{
    class Program
    {
        static void Main()
        {
            var cnn = new CNN();

            Console.WriteLine("Loading the model for prediction...");

            Console.WriteLine("Making a prediction...");
            var sampleData = new DummyData { Feature1 = 0.7f, Feature2 = 0.4f };
            var prediction = cnn.Predict(sampleData);

            Console.WriteLine($"Predicted label: {prediction}");
        }
    }
}
