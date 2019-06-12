package de.rif.ai.textanalyzer;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class ExtractingRelationship {

	public static void main(String[] args) {
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, parse");
		
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		
		Annotation annotation = new Annotation("The meaning and purpose of life is plain to see.");
		pipeline.annotate(annotation);
		pipeline.prettyPrint(annotation, System.out);
	}

}
