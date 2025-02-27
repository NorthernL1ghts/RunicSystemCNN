# RunicSystemCNN

A simple C# program using ML.NET to create, train, and use a basic machine learning model for classification.

## Concept: Rune-Based Knowledge System
This project serves as a proof of concept (POC) for a framework based on magical runes for a multiplayer game. The inspiration comes from spell books—traditionally used to share knowledge. Instead of physically acquiring runes, a player would "know" a rune and be able to use it instantly. However, to avoid instant discovery of all runes, a research aspect is introduced.

### Research and Discovery
- Players study runes, gradually discovering and perfecting them.
- A convolutional neural network (CNN) is used to detect drawn runes.
- The CNN outputs both the detected rune and its confidence level.
- The confidence level amplifies the rune's effect in-game.
- Players can iteratively refine their runes to optimize their effectiveness.

For example, a player may find that adding a small line enhances a rune's effect. Over time, they refine their designs, making their runes more potent.

### Credit
Concept suggested by **Nothavid**. The CNN implementation helps save time and streamline rune detection and enhancement.

## Features
- Generates dummy data with two numerical features and a binary label.
- Preprocesses data using feature concatenation and normalization.
- Uses the SDCA (Stochastic Dual Coordinate Ascent) algorithm for multi-class classification.
- Saves and loads the trained model.
- Makes predictions on new data.

## Requirements
- .NET 9 or later
- ML.NET NuGet package

## Installation
1. Clone the repository or download the `Program.cs` file.
2. Ensure you have .NET installed on your system.
3. Install the required ML.NET package if not already installed:
   ```sh
   dotnet add package Microsoft.ML
   ```

## Usage
1. Build and run the program:
   ```sh
   dotnet run
   ```
2. The program will:
   - Generate and load dummy data.
   - Train a machine learning model.
   - Save the trained model to `model.zip`.
   - Load the model and make a prediction.
   - Print the predicted label to the console.

## Example Output
```
Generating dummy data...
Loading dummy data...
Defining the pipeline...
Training the model...
Saving the model...
Loading the model for prediction...
Making a prediction...
Predicted label: 1
```

## Notes
- This example demonstrates a simple ML.NET pipeline for classification.
- The model uses only two features, but it can be extended for more complex datasets.

## License
This project is licensed under the MIT License and is available for modification and distribution.
