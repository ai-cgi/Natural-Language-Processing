package de.rif.ai.textanalyzer;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.StringReader;
import java.util.Iterator;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordTokenFactory;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;

public class NlpTokenizer {

	public static void main(String[] args) {
		String paragraph = "Let's pause, \nand then reflect.";
		
		printSimpleTokenizer(paragraph);
		
		printWhitespaceTokenizer(paragraph);
		
		printTokenizerME(paragraph);
		
		printPtbTokenizer(paragraph);
		
		printDocumentPreprocessor(paragraph);
	}

	private static void printDocumentPreprocessor(String paragraph) {
		Reader reader = new StringReader(paragraph);
		DocumentPreprocessor docPreprocessor = new DocumentPreprocessor(reader);
		
		Iterator<List<HasWord>> it = docPreprocessor.iterator();
		System.out.println("DocumentPreprocessor");
		while (it.hasNext()) {
			List<HasWord> sentence = it.next();
			for (HasWord token : sentence) {
				System.out.println(token);
			}
		}
	}

	private static void printPtbTokenizer(String paragraph) {
		@SuppressWarnings("unchecked")
		PTBTokenizer ptb = new PTBTokenizer(new StringReader(paragraph), new CoreLabelTokenFactory(), null);
		System.out.println("PTBTokenizer - CoreLabelTokenFactory");
		while (ptb.hasNext()) {
			System.out.println(ptb.next());
		}
		
		ptb = new PTBTokenizer(new StringReader(paragraph), new WordTokenFactory(), null);
		System.out.println("PTBTokenizer - WordTokenFactory");
		while (ptb.hasNext()) {
			System.out.println(ptb.next());
		}
		
	}

	private static void printTokenizerME(String paragraph) {
		try {
			InputStream modelIn = new FileInputStream("./src/main/resources/data/en-token.bin");
			
			TokenizerModel model = new TokenizerModel(modelIn);
			Tokenizer tokenizer = new TokenizerME(model);
			String tokens[] = tokenizer.tokenize(paragraph);
			System.out.println("TokenizerME");
			for (String s : tokens) {
				System.out.println(s);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void printWhitespaceTokenizer(String paragraph) {
		String tokens[] = WhitespaceTokenizer.INSTANCE.tokenize(paragraph);
		System.out.println("WhitespaceTokenizer");
		for (String s : tokens) {
			System.out.println(s);
		}
	}

	private static void printSimpleTokenizer(String paragraph) {
		SimpleTokenizer simpleTokenizer = SimpleTokenizer.INSTANCE;
		String tokens[] = simpleTokenizer.tokenize(paragraph);
		System.out.println("SimpleTokenize");
		for (String s : tokens) {
			System.out.println(s);
		}
	}

}
