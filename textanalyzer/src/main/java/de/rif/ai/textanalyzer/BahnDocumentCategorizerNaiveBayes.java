package de.rif.ai.textanalyzer;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.ml.AbstractTrainer;
import opennlp.tools.ml.naivebayes.NaiveBayesTrainer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

public class BahnDocumentCategorizerNaiveBayes {

	public static void main(String[] args) {

		try {
			// Read the training data file.
			InputStreamFactory dataIn = new MarkableFileInputStreamFactory(
					new File("./src/main/resources/data/train.txt"));
			ObjectStream<String> lineStream = new PlainTextByLineStream(dataIn, "UTF-8");
			ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

			// Define the training parameters.
			TrainingParameters params = new TrainingParameters();
			params.put(TrainingParameters.ITERATIONS_PARAM, 10000 + "");
			params.put(TrainingParameters.CUTOFF_PARAM, 0 + "");
			params.put(AbstractTrainer.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE);

			// Train and create a model from the training data and defined training
			// parameters.
			DoccatModel model = DocumentCategorizerME.train("de", sampleStream, params, new DoccatFactory());

			// Save the newly trained model to a local file, which can be used later for
			// predicting movie genere.
			BufferedOutputStream modelOut = new BufferedOutputStream(
					new FileOutputStream("./src/main/resources/data/bahn-classifier-naive-bayes.bin"));
			model.serialize(modelOut);

			// Test the model for a sample string and print the probabilities for the string
			// to belong to different categories. The method
			// DocumentCategorizer.categorize(String[] wordsOfDoc) takes words of a document
			// as an argument in the form of an array of Strings.
			DocumentCategorizer doccat = new DocumentCategorizerME(model);
			double[] aProbs = doccat.categorize(
					"Brühl-Vochem (KBRV): die Fahrzeit auf der eingleisigen Strecke zwischen KBRV und KBR G muss in beide Richtungen 4 Minuten betragen -> Zusatzhalt vor/nach KBR G um die Streckenkapazität richtig darzustellen"
							.replaceAll("[^A-Za-z]", " ").split(" "));

			// print the probabilities of the categories
			System.out.println(
					"\n---------------------------------\nCategory : Probability\n---------------------------------");
			for (int i = 0; i < doccat.getNumberOfCategories(); i++) {
				System.out.println(doccat.getCategory(i) + " : " + aProbs[i]);
			}
			System.out.println("---------------------------------");

			System.out.println(
					"\n" + doccat.getBestCategory(aProbs) + " : is the predicted category for the given sentence.");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
