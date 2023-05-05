from project3 import readPdf

# Check if the pdf is being read correctly
def test_ReadPdf(contentsPath = 'smartcity/AZ Tucson.pdf'): 
    
    content = readPdf("AZ Tucson", contentsPath)
    
    # Check if the different rows are being combined as 1
    assert len(content) == 1

    
    