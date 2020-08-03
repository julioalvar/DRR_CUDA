#include "mlDRR_CUDA.h"

// Include tools for accessing the DICOM tree
#include <mlDicomTreeAccess.h>
#include <DCMTree_Utils.h>

ML_START_NAMESPACE


void PROJECT_CLASS_NAME::readDICOMTagFromInputImage(unsigned inputNumber, Vector3d &ctPixelSpacing)
{
	string tagName_PixelSpacing = "PixelSpacing";
	string tagName_Rows = "Rows";
	string tagName_Columns = "Columns";
	string tagName_NumberOfFrames = "NumberOfFrames";
	string tagName_SpacingBetweenSlices = "SpacingBetweenSlices";

	string tagValue_PixelSpacing("");
	string tagValue_Rows("");
	string tagValue_Columns("");
	string tagValue_NumberOfFrames("");
	string tagValue_SpacingBetweenSlices("");

	std::string tagValue("");
	std::string tagVR("");
	std::vector < std::string > statusMessages;



	PagedImage* pagedImage = getUpdatedInputImage(inputNumber);

	if (pagedImage)
	{

		if (readDICOMTagFromInputImage(pagedImage, tagName_PixelSpacing, tagValue_PixelSpacing, tagVR, statusMessages))
		{
			size_t backslash = tagValue_PixelSpacing.find("\\");
			if (backslash != std::string::npos)
			{
				string x_component = tagValue_PixelSpacing.substr(0, backslash);
				string y_component = tagValue_PixelSpacing.substr(backslash + 1);
				ctPixelSpacing[2] = stod(x_component);
				ctPixelSpacing[1] = stod(y_component);

				//cout << tagName_PixelSpacing << ": " << x_component << " \\ " << y_component << endl;
			}
		}

		/*if (readDICOMTagFromInputImage(pagedImage, tagName_Rows, tagValue_Rows, tagVR, statusMessages))
		{
			//cout << tagName_Rows << ": " << tagValue_Rows << endl;
			dimY = stoi(tagValue_Rows);
		}

		if (readDICOMTagFromInputImage(pagedImage, tagName_Columns, tagValue_Columns, tagVR, statusMessages))
		{
			//cout << tagName_Columns << ": " << tagValue_Columns << endl;
			dimX = stoi(tagValue_Columns);
		}

		if (readDICOMTagFromInputImage(pagedImage, tagName_NumberOfFrames, tagValue_NumberOfFrames, tagVR, statusMessages))
		{
			//cout << tagName_NumberOfFrames << ": " << tagValue_NumberOfFrames << endl;
			dimZ = stoi(tagValue_NumberOfFrames);
		}*/

		if (readDICOMTagFromInputImage(pagedImage, tagName_SpacingBetweenSlices, tagValue_SpacingBetweenSlices, tagVR, statusMessages))
		{
			//cout << tagName_SpacingBetweenSlices << ": " << tagValue_SpacingBetweenSlices << endl;
			ctPixelSpacing[0] = stod(tagValue_SpacingBetweenSlices);
		}
	}


}


bool PROJECT_CLASS_NAME::readDICOMTagFromInputImage(PagedImage* pagedImage, const std::string& tagName, std::string& tagValue, std::string& tagVR, std::vector < std::string >& statusMessages)
{
	const RuntimeType*            dicomPropRuntimeType = DicomTreeImagePropertyExtension::getClassTypeId();
	const ImagePropertyExtension* inputImageProperties = pagedImage->getImagePropertyContainer().getFirstEntryOfType(dicomPropRuntimeType);

	DCMTree::Const_TreePtr dicomTreePtr = getDicomTreeFromImagePropertyExtension(inputImageProperties);

	if (dicomTreePtr)
	{
		DCMTree::Const_TagPtr tagPtr;
		try
		{
			tagPtr = dicomTreePtr->getTag(tagName);
		}
		catch (DCMTree::Exception&)
		{
			statusMessages.push_back("DICOM tag with the given name not found.");
		}
		if (!tagPtr)
		{
			// maybe the tag's name is really unknown, or it is coded in another format
			bool                   tagIdIsValid = false;
			DCMTree::TagId         tagId;
			DCMTree::Const_DictPtr dictPtr = dicomTreePtr->getDict();
			if (!dictPtr)
			{
				// Fall back to default dictionary singleton if tree does not carry its own dictPtr.
				dictPtr = DCMTree::Dict::singleton();
			}

			if (dictPtr)
			{
				if (dictPtr->isTagNameKnown(tagName))
				{
					try
					{
						tagId = dictPtr->tagId(tagName);
						tagIdIsValid = true;
					}
					catch (DCMTree::Exception&)
					{
						statusMessages.push_back("Unknown tag name.");
					}
				}
			}
			else
			{
				statusMessages.push_back("No DICOM dictionary available.");
			}

			if (!tagIdIsValid)
			{
				// try a different format for the tag name
				if (MLIsATString(tagName))
				{
					tagIdIsValid = getTagIdByATString(tagName, tagId);
					if (!tagIdIsValid)
					{
						statusMessages.push_back("Failed to get tag id by AT string.");
					}
				}
				else
				{
					statusMessages.push_back("No valid AT string given.");
				}
			}
			if (tagIdIsValid)
			{
				tagPtr = dicomTreePtr->getTag(tagId);
			}
		}

		if (tagPtr)
		{
			// set value if value has changed
			DCMTree::Vr vr = tagPtr->info().vr();

			if (!(DCMTree::isType(tagPtr->info().vr(), DCMTree::TY_Bin)))
			{
				try
				{
					tagValue = tagPtr->toString();
				}
				catch (DCMTree::Exception&)
				{
					statusMessages.push_back("Could not serialize tag value.");
				}

				try
				{
					tagVR = DCMTree_Utils::toString(vr);
				}
				catch (DCMTree::Exception&)
				{
					statusMessages.push_back("Could not read VR.");
				}
			}
		}
	}
	else
	{
		statusMessages.push_back("No DICOM tree available.");
	}
	return statusMessages.size() == 0;
}


ML_END_NAMESPACE