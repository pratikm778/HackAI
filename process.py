from embeddings_processor import EmbeddingsProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize the processor
        processor = EmbeddingsProcessor()
        
        # Process both text and images
        logger.info("Starting data processing...")
        processor.process_data_folder(data_folder="data", image_folder="pic_data")
        logger.info("Data processing completed!")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise

if __name__ == "__main__":
    main()