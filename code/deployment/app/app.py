import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets.preprocess import preprocess_data
import json
import requests
import pandas as pd
import re
from datetime import datetime

port_number = 5001

current_year = datetime.now().year
years = list(range(current_year - 100, current_year + 1))
months = [f"{i:02d}" for i in range(0, 12)]

# Define the predict function
def predict(month=None,
            town=None,
            flat_type=None,
            block=None,
            street_name=None,
            storey_range=None,
            floor_area_sqm=None,
            flat_model=None,
            lease_commence_date=None,
            remaining_lease_years=None,
            remaining_lease_months=None):

    # Combine years and months into the desired format
    remaining_lease = f"{int(remaining_lease_years)} years {int(remaining_lease_months):02d} months"
    
    # Create features dictionary
    features = {
        "month": month,
        "town": town,
        "flat_type": flat_type,
        "block": block,
        "street_name": street_name,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area_sqm,
        "flat_model": flat_model,
        "lease_commence_date": lease_commence_date,
        "remaining_lease": remaining_lease,
    }
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # Transform the input data
    prep_path = 'data/prep.pkl'
    X, _ = preprocess_data(data=raw_df, prep_path=prep_path, only_X=True)
    
    # Convert to JSON
    example = X.iloc[0]
    example = json.dumps(example.to_dict())
    print('Example', example)

    payload = example

    # Send POST request with the payload to the deployed Model API
    response = requests.post(
        url=f"http://flask-api:{port_number}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response)
    
    return response.json()

def validate_month_format(month):
    pattern = re.compile(r'^\d{4}-\d{2}$')
    if pattern.match(month):
        return True, ""
    else:
        return False, "Invalid format. Please enter the date in YYYY-MM format."

# Define Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Month", placeholder="YYYY-MM"), 
        gr.Dropdown(label="Town", choices=['SEMBAWANG', 'PUNGGOL', 'BUKIT BATOK', 'TOA PAYOH', 'QUEENSTOWN',
       'TAMPINES', 'YISHUN', 'CLEMENTI', 'WOODLANDS', 'HOUGANG',
       'JURONG WEST', 'PASIR RIS', 'BUKIT PANJANG', 'BUKIT MERAH',
       'BEDOK', 'CHOA CHU KANG', 'SENGKANG', 'KALLANG/WHAMPOA',
       'ANG MO KIO', 'SERANGOON', 'CENTRAL AREA', 'JURONG EAST', 'BISHAN',
       'GEYLANG', 'MARINE PARADE', 'BUKIT TIMAH']),
        gr.Dropdown(label="Flat Type", choices=['3 ROOM', '5 ROOM', '4 ROOM', 'EXECUTIVE', '2 ROOM', '1 ROOM',
       'MULTI-GENERATION']),
        gr.Dropdown(label="Block", choices=['590B', '128A', '182', '138B', '153', '843', '458', '416', '622', '352', '180C', '540', '503', '138A', '510', '684', '136', '873', '574', '130', '20', '215', '738', '557', '663C', '890A', '516', '246', '603', '649B', '208B', '298', '759', '139B', '6', '210A', '817B', '249', '469', '3', '62', '70A', '816B', '842C', '508', '211', '276', '488B', '426', '686B', '34', '338', '343', '112', '344', '314B', '334C', '105', '349', '171', '304B', '298B', '4', '101', '560', '235', '525', '874', '601B', '110', '30', '196B', '311B', '289B', '65', '433', '28', '178D', '667', '661A', '3C', '641', '679A', '315C', '50', '810A', '730', '535A', '213', '184', '314A', '225', '612', '408', '166', '630', '309', '79B', '357', '286', '755', '75B', '522', '120', '229', '348A', '824', '281A', '596B', '295', '205', '750', '662', '290G', '389', '234', '624A', '395', '329', '727', '1', '21', '664', '103', '330', '120B', '525C', '703', '315', '173', '333A', '477B', '542', '17', '163', '475A', '230F', '180', '692B', '455', '367', '689', '147', '5', '489D', '614B', '548B', '709', '519C', '619', '292B', '551', '2', '115', '870', '245', '588', '415', '197', '636A', '239', '774', '450', '898', '529', '233', '31', '514', '553', '435', '871', '365B', '546A', '350', '422', '619D', '223A', '135', '441A', '18', '593B', '725', '116B', '549A', '131', '485', '767', '805D', '36', '768', '271A', '15', '440A', '354D', '633', '178B', '160B', '48', '846', '605', '130A', '10A', '123', '163B', '850', '183A', '251', '66', '981C', '615', '418', '444', '505', '124', '940', '121', '53', '603B', '476B', '816', '59', '671B', '606', '663B', '603A', '143', '102B', '947', '95C', '436', '714', '336C', '273D', '26', '89', '278', '193', '363B', '54', '681D', '108', '130B', '651A', '546C', '530C', '469A', '43A', '512B', '28B', '443A', '804B', '170', '232A', '678D', '62A', '151', '673A', '854', '404A', '265D', '113', '107B', '173B', '275B', '267A', '504', '443B', '929', '79E', '845', '218', '495C', '109', '708', '676', '498D', '183', '460A', '521', '433B', '1B', '14A', '669B', '702', '288B', '121D', '673C', '161', '297', '188', '3B', '311C', '263', '640', '307A', '256', '192A', '673B', '386', '787', '456', '782A', '44', '303C', '243', '49', '786B', '588C', '694C', '932', '536', '651', '254', '307', '416B', '264E', '707', '639', '206', '661', '301', '611', '201', '169', '322', '471', '817', '535', '453C', '35', '753', '440C', '507', '365C', '618', '10', '207C', '110A', '117C', '326', '258', '106', '522B', '274A', '788B', '126', '94', '861', '451', '274B', '270', '140', '282B', '102C', '626', '104B', '227', '506', '154', '308', '488C', '481', '16', '217B', '467A', '670', '63', '664A', '812A', '19', '122', '454', '694D', '336D', '266', '157', '228A', '141', '325', '420', '987A', '223', '122D', '807B', '288A', '404', '1F', '362', '608', '118', '584', '80A', '214', '194B', '85', '273A', '686A', '342B', '166A', '131A', '315B', '621', '264', '602C', '638', '780C', '472', '206A', '443D', '322A', '409B', '513', '571A', '272B', '51', '180B', '126A', '100', '320D', '109B', '666B', '348C', '690E', '471A', '643', '453', '87', '185', '470A', '780A', '877', '486', '476C', '453D', '604A', '337', '889A', '874A', '191', '24', '423', '346', '533', '129B', '278C', '616', '705', '58', '417', '285', '32', '554', '363', '668B', '432A', '674A', '717', '728', '812', '790', '592C', '308B', '64', '289', '403', '318B', '217', '578', '546', '158B', '464', '310C', '351A', '269', '265C', '612A', '609', '149', '783B', '624', '242', '174D', '989C', '208', '128', '537', '206D', '489B', '230', '238', '629', '196D', '606C', '188A', '320', '649', '784', '134', '366', '326C', '627', '691B', '104', '61', '124C', '295C', '402', '414', '309D', '526D', '325B', '686C', '722', '125', '177', '236', '818', '436A', '683B', '119D', '116', '321', '139A', '174', '716', '549', '255', '1D', '331A', '418A', '342C', '415C', '250B', '272A', '498J', '409', '41', '291A', '818C', '1C', '772', '646', '763', '685C', '527', '168D', '327C', '38A', '452A', '204', '37', '224C', '81', '487', '442B', '443C', '305', '78', '538', '981A', '921', '371', '499A', '77', '216', '156', '250', '618D', '445A', '547C', '411B', '12C', '885', '220', '431', '111', '119B', '547', '894A', '424', '69', '591A', '412', '656A', '676B', '140B', '316', '636B', '83', '273B', '289A', '476A', '373', '575', '696', '138D', '475B', '270A', '407', '455C', '86', '302B', '90', '202A', '506A', '220C', '201C', '302', '666A', '160', '704', '734', '439C', '765', '732', '168B', '984A', '121B', '332A', '447', '354', '812B', '28C', '756', '633C', '466A', '974', '119', '106B', '209', '11', '93B', '9', '268C', '493E', '164', '550A', '2C', '467', '253', '209A', '285D', '333D', '509C', '286D', '461', '580', '478', '210', '98', '195', '302A', '327', '38B', '771', '216B', '217A', '231B', '110D', '144', '167A', '992A', '848', '181', '659B', '306B', '271', '7', '91', '121C', '460', '76', '350A', '497J', '868', '40', '317C', '333B', '168', '43', '720', '12', '889B', '145', '448C', '427', '23', '268D', '610', '47', '312B', '761', '241', '855', '979A', '677B', '28A', '370', '306C', '172', '114', '1E', '341', '926', '25', '676C', '770', '746', '449', '226A', '247', '439B', '507B', '332', '424C', '429', '259', '180A', '678A', '189', '690D', '384', '569', '670C', '526', '42', '530', '541', '958', '396', '187', '324', '142', '46', '656C', '517', '410C', '334', '689B', '663A', '638C', '468C', '778', '281', '197D', '419', '803A', '943', '534', '22', '273', '721', '226', '293B', '782C', '8', '226C', '528A', '111A', '474', '127D', '146', '501B', '139', '280', '5A', '587', '917', '555', '228', '641C', '607', '60', '317', '511', '729', '902', '330A', '511A', '232', '857B', '672B', '133', '55', '165B', '723', '518', '368', '26B', '887A', '432B', '8B', '155', '809', '567', '276B', '288D', '669', '257', '672A', '448A', '202', '432', '457', '266C', '365D', '339A', '615A', '285B', '980B', '512A', '67', '687B', '266D', '783A', '61C', '115A', '528B', '706', '92B', '886A', '372', '563', '604', '735', '224', '6A', '736', '894C', '8A', '426B', '617C', '480', '13', '306', '211C', '463C', '299', '312C', '316A', '658C', '296B', '232C', '310B', '518C', '138', '391', '118C', '120D', '121A', '233C', '172B', '520B', '179', '865', '307C', '309B', '117', '462', '109C', '674', '267', '435C', '331', '925', '441', '602', '284', '559', '571', '33', '169C', '625', '119C', '4C', '892A', '137', '434A', '222', '297C', '366A', '451B', '355', '345', '335', '162', '636C', '288C', '286C', '45', '91A', '869A', '308A', '593A', '195A', '665', '175B', '623', '103B', '437', '63B', '617A', '475', '936', '93', '520', '92', '93A', '413', '566', '619B', '886B', '405B', '613D', '207B', '524', '313', '690B', '348B', '424B', '987C', '186C', '491', '105A', '662B', '415A', '766', '287', '469B', '185C', '944', '435A', '802', '548', '670B', '131C', '72', '572', '797', '550', '890C', '150A', '61A', '310A', '138C', '336', '988A', '991A', '186', '802A', '471B', '339', '159', '539', '570', '327A', '167', '617B', '88', '272D', '669A', '473', '801C', '648', '652C', '712', '509', '365A', '808', '288', '353B', '351C', '550B', '312', '476', '304', '271C', '185D', '851', '9B', '654', '815B', '448', '95', '2D', '406B', '462D', '213A', '980', '10C', '200', '788E', '438B', '240', '453B', '199', '231', '836', '601A', '688A', '564', '417A', '660D', '624C', '359C', '931', '596A', '71', '803B', '631', '528', '813', '394', '217C', '90A', '666', '505D', '796', '896A', '221', '353', '335A', '196A', '334B', '74', '659D', '635B', '561', '465', '784B', '614A', '351', '524A', '122A', '810', '543', '625B', '497C', '94B', '677', '313B', '986A', '198', '532', '642', '150', '544', '477', '94C', '545', '502', '832', '258B', '523B', '897B', '623B', '286A', '113B', '421', '680A', '502D', '276C', '335B', '388', '440', '671A', '39', '801A', '322B', '764', '742', '326D', '411', '38', '328', '496G', '504D', '18B', '425', '635', '726', '108B', '165', '448B', '966', '190A', '175D', '459', '303B', '25A', '999B', '29', '310', '126B', '412A', '585', '817A', '501', '497D', '152', '132', '470', '107', '910', '450A', '316C', '657A', '783C', '348D', '604C', '381', '655', '811A', '786F', '785D', '311', '634', '451A', '169B', '219', '655A', '301C', '408B', '14', '504C', '212', '815', '226B', '102A', '102', '879', '27A', '186A', '769', '673', '684C', '212A', '596C', '996B', '699', '676A', '739', '210B', '688E', '636', '188C', '483', '692A', '301B', '519', '826', '375', '654B', '52', '441B', '359', '314D', '664B', '293D', '259A', '749', '515B', '613', '490D', '957', '339D', '118D', '834', '476D', '888', '694B', '79A', '701', '211A', '452', '844', '443', '157A', '751', '685A', '658', '18A', '791', '272', '265', '68', '410A', '252', '297D', '576', '208C', '377', '410', '470B', '296A', '317D', '264F', '314', '210C', '318C', '261', '668', '688', '988B', '410B', '647', '170B', '338D', '466C', '512C', '172C', '217D', '358', '275', '695', '662A', '787D', '262', '280B', '556', '401', '334D', '129', '176', '618B', '804', '588D', '490B', '502A', '637B', '127', '406', '488D', '524B', '122B', '324C', '762', '635A', '186B', '438', '393', '747B', '494D', '932A', '969', '894D', '445', '878', '711', '508A', '70', '248', '10D', '724', '127C', '852', '333', '244', '674B', '693', '356', '644', '808A', '805A', '661B', '491E', '484', '194A', '348', '462A', '406A', '317A', '228B', '719', '785B', '691A', '262B', '733', '887', '896', '780F', '632', '17B', '319A', '572A', '336B', '648A', '265A', '382', '508B', '361', '337A', '660', '390', '489', '175A', '213B', '2B', '879A', '687C', '211B', '505B', '689A', '442', '107A', '201B', '187B', '899C', '430B', '998B', '827', '830', '296E', '547D', '299B', '904', '282C', '291E', '787E', '312A', '164C', '526A', '662C', '305D', '274', '487C', '216A', '462B', '260A', '492', '208A', '304A', '118A', '558', '338C', '362B', '889C', '515C', '290E', '315A', '171A', '914', '183C', '271D', '471C', '158', '10B', '406C', '303A', '148', '748C', '434B', '264A', '1G', '833', '710', '886C', '195E', '445B', '233B', '128D', '472C', '526B', '455A', '491C', '498M', '441C', '665A', '775', '77A', '789', '221A', '99', '323B', '653', '27B', '655B', '546B', '512', '869', '570B', '1A', '287B', '604B', '429A', '752', '520A', '984C', '989D', '660C', '70B', '884', '256A', '950', '70C', '579', '160A', '527C', '613A', '8C', '995A', '197A', '463', '691D', '665B', '731', '661C', '331B', '212B', '335C', '111B', '867', '303', '592B', '4B', '653A', '308C', '275D', '590A', '171C', '405C', '337D', '815C', '583', '860', '75A', '805', '978C', '612D', '468', '493B', '439', '446', '18D', '165A', '294A', '842H', '110B', '743', '115B', '628', '442C', '741', '107C', '686', '671', '875', '313C', '414A', '798', '839', '176A', '313D', '23B', '978', '333C', '319', '75', '356B', '523D', '601D', '757', '979B', '573C', '473B', '677A', '645', '322D', '106A', '201D', '520C', '561A', '258A', '430', '298D', '522A', '250A', '104A', '472B', '675C', '237', '475D', '10F', '178A', '912', '257A', '293A', '209C', '718', '194', '777', '491H', '488A', '509A', '615B', '3A', '672C', '748A', '482', '336A', '493C', '603C', '653C', '625A', '436B', '780E', '97', '289D', '840', '357A', '638A', '806', '847', '934', '823', '683C', '552', '494G', '197B', '262D', '129C', '933A', '207', '199B', '970', '392', '9A', '131B', '195B', '805B', '938', '428', '530D', '678', '360C', '811', '297B', '659C', '80B', '684B', '501A', '292A', '808B', '17A', '588B', '432D', '678C', '713', '140D', '687D', '942', '411A', '162B', '498G', '570A', '408A', '515', '120A', '842F', '890B', '383', '447A', '220B', '473D', '681', '659', '291B', '758', '108C', '878A', '293', '899B', '446B', '680', '987B', '747', '782E', '57', '531', '306A', '491D', '828', '116A', '332B', '498E', '525A', '693B', '302D', '360A', '430A', '2A', '256D', '291', '73', '601C', '989A', '979C', '455B', '403B', '561B', '990C', '547A', '783D', '409A', '642D', '690', '794', '489A', '683', '685B', '338A', '85B', '153A', '279C', '103C', '223D', '201A', '841', '667B', '681C', '387', '473A', '492G', '163A', '808D', '637', '80C', '863', '256C', '527D', '275A', '185B', '484D', '507A', '623C', '995C', '469C', '318', '56', '272C', '440B', '699B', '332C', '59B', '287A', '897A', '858', '787C', '104C', '182A', '650B', '286B', '667C', '326A', '504B', '989B', '117B', '438A', '505C', '323', '442A', '203A', '657B', '434', '158C', '176B', '109A', '279', '190', '123A', '907', '523C', '612C', '988C', '573B', '203', '319C', '867A', '234A', '802B', '565', '167B', '688D', '353A', '397', '167C', '574B', '118B', '191B', '268', '291C', '173A', '189C', '80', '213C', '278B', '915', '682B', '601', '279B', '101C', '634B', '374', '492C', '405', '12B', '119A', '596D', '490', '684A', '893', '923', '697A', '128C', '924', '379', '573', '913', '313A', '261B', '820', '466', '809A', '487B', '113A', '323A', '494B', '240A', '403C', '829', '978D', '754', '418B', '831', '498A', '687', '200B', '984D', '782B', '101A', '681B', '414B', '84', '430D', '307B', '477A', '870A', '495F', '689E', '76A', '273C', '869B', '697', '329B', '849', '895A', '268B', '620', '530B', '266B', '560A', '880', '896C', '259C', '613B', '518B', '893A', '842', '205A', '853', '337C', '325A', '871B', '233A', '282', '780B', '168A', '404B', '6B', '453A', '122E', '270B', '490C', '269A', '28D', '181A', '682A', '661D', '935', '203D', '357B', '274C', '571C', '675A', '489C', '192C', '299C', '955', '351B', '277B', '360', '807', '946', '159A', '783', '479', '868A', '224B', '612B', '605C', '362C', '670A', '204B', '568', '682C', '290F', '493', '197C', '973', '776', '922', '889D', '431B', '648B', '951', '650', '269D', '687A', '491G', '487A', '689D', '30A', '997C', '615C', '811B', '323C', '347', '309A', '38C', '116C', '894', '737', '365', '178C', '814', '260', '589B', '501C', '468A', '426C', '987D', '698B', '688B', '467B', '301A', '815A', '267B', '339B', '297A', '498H', '623A', '305A', '658B', '352C', '267C', '340', '549B', '428B', '289G', '187A', '856C', '577', '882', '998A', '81B', '295B', '972', '527B', '570C', '350C', '981D', '683D', '183D', '519A', '178', '75C', '588A', '234B', '96', '206C', '283', '271B', '895C', '311D', '523', '475C', '807C', '281B', '657', '683A', '413A', '488', '648C', '740', '941', '698', '290C', '364', '350B', '261A', '694A', '431A', '61B', '690A', '450B', '218A', '887B', '27', '504A', '513D', '26D', '305C', '590C', '908', '424A', '227B', '690F', '278A', '305B', '888A', '909', '204D', '225A', '801B', '613C', '264C', '782D', '277', '589', '428A', '450E', '614', '158D', '3D', '795', '470C', '95B', '440D', '95A', '663', '219C', '274D', '698A', '838', '495E', '206B', '773', '461C', '407B', '413B', '63A', '684D', '295A', '939', '224A', '449B', '277C', '491A', '678B', '468B', '986B', '813B', '862', '807A', '525B', '513B', '748', '692', '175C', '426A', '296D', '622B', '280A', '501D', '207D', '818A', '264D', '439A', '450D', '202C', '353C', '658A', '288G', '124A', '289C', '79C', '269C', '317B', '342', '519B', '932B', '562', '819', '802C', '688C', '316B', '285C', '808C', '261D', '690C', '117A', '301D', '960', '883', '805C', '700', '675B', '691', '167D', '497G', '168C', '450C', '672D', '296', '62B', '227C', '257B', '157C', '503C', '693C', '311A', '463B', '376A', '682D', '682', '124B', '462C', '665C', '269B', '291D', '276A', '415B', '260B', '298C', '903', '223B', '898A', '889', '472A', '183B', '634A', '511B', '586', '786D', '837', '299A', '266A', '174B', '265E', '606B', '637A', '494J', '677C', '662D', '189A', '416C', '4A', '652', '744', '466B', '351D', '804A', '493D', '203B', '980C', '856F', '694', '697B', '548A', '175', '82', '203E', '976', '324D', '218B', '294', '140A', '668C', '354A', '638B', '835', '211D', '327B', '461A', '196', '928', '113D', '788C', '176D', '473C', '296C', '589D', '510B', '953', '306D', '466D', '435B', '162C', '781', '215A', '656', '952', '468D', '220A', '650A', '222A', '196C', '321A', '126D', '786E', '984B', '782', '450G', '166B', '126C', '857', '628A', '307D', '181B', '200D', '288E', '592A', '38D', '18C', '641A', '26A', '85A', '96A', '59A', '617D', '677D', '891', '618A', '871C', '764A', '450F', '485A', '633B', '818B', '497H', '129A', '689C', '260C', '192', '571B', '446C', '188B', '102D', '617', '872', '268A', '494C', '170A', '979', '322C', '780D', '787B', '484B', '426D', '26C', '319B', '628B', '503B', '497F', '886D', '161A', '679', '698C', '887C', '115C', '287C', '340B', '842G', '292', '715', '12A', '700A', '219A', '498F', '347B', '497A', '859', '991B', '164A', '338B', '285A', '685', '505A', '293C', '897C', '262C', '189B', '74A', '492D', '270C', '801D', '170C', '289E', '431C', '431D', '204A', '785', '369', '821', '318A', '788', '619C', '446A', '803', '648D', '992B', '323D', '84B', '416A', '879B', '920', '899', '876', '265B', '649A', '878B', '218C', '79', '231A', '500', '498B', '205C', '112A', '385', '659A', '502C', '107D', '359B', '277D', '257C', '433A', '618C', '574A', '496E', '589C', '318D', '964', '158A', '967', '686D', '519D', '261C', '633D', '485C', '809B', '667A', '660A', '366B', '997B', '403D', '399', '486A', '658D', '182B', '104D', '424D', '589A', '180D', '39A', '212C', '515D', '748B', '205B', '230H', '221B', '786', '813A', '342A', '436C', '526C', '911', '83B', '981B', '292C', '891B', '163C', '681A', '258C', '430C', '622C', '127A', '513C', '185A', '803D', '432C', '784C', '195D', '325C', '895', '173D', '326B', '486C', '184B', '937', '376B', '675', '517A', '494H', '632B', '494E', '502B', '891A', '184A', '803C', '108A', '216C', '513A', '161B', '830A', '977', '760', '92A', '881', '320C', '347A', '945', '633A', '334A', '916', '530A', '497B', '792', '933', '930', '324A', '679C', '602B', '985A', '29A', '962', '164B', '668A', '637C', '300', '227A', '971', '898B', '817C', '330B', '441D', '671C', '485B', '745', '899A', '527A', '799', '517C', '893C', '25B', '496B', '209B', '518A', '786C', '113C', '747A', '698D', '123B', '816A', '664C', '506B', '225C', '23A', '523A', '495D', '357C', '171B', '207A', '641B', '436D', '985B', '259B', '219D', '656D', '321B', '498L', '693A', '331C', '162A', '892B', '140C', '896B', '986C', '624B', '101D', '672', '495', '825', '376', '250D', '996C', '895B', '688F', '437A', '429B', '919', '864', '230C', '669D', '122C', '157D', '101B', '573A', '654C', '667D', '123D', '717A', '507C', '452B', '700B', '258D', '202B', '619A', '447B', '652A', '642B', '363A', '856', '517D', '376C', '256B', '980D', '927', '173C', '298A', '359A', '506C', '642A', '289F', '264B', '199C', '282A', '223C', '842D', '204C', '747C', '320A', '200A', '188D', '176C', '602A', '215B', '356C', '963', '490A', '442D', '340A', '656B', '354C', '302C', '115D', '250C', '605A', '965', '511C', '700C', '219B', '495A', '493A', '314C', '507D', '157B', '676D', '663D', '290', '499B', '581', '408C', '190B', '218D', '897', '262A', '862A', '496D', '236A', '518D', '522C', '864A', '232B', '405A', '968', '918', '632A', '784A', '103A', '484A', '329A', '572B', '316D', '642C', '352A', '48A', '606D', '230B', '622A', '128B', '509B', '508C', '779', '106D', '79D', '303D', '637D', '699C', '654A', '172A', '360B', '449A', '461D', '109D', '528C', '660B', '980A', '96B', '691C', '364A', '199A', '174A', '515A', '288F', '106C', '812C', '796A', '287D', '228C', '463A', '679B', '632C', '842B', '485D', '668D', '82B', '341B', '59C', '123C', '605B', '547B', '352B', '279A', '90B', '10E', '412B', '275C', '309C', '990A', '664D', '290A', '859A', '169A', '85C', '606A', '105C', '975', '822', '105B', '460C', '73A', '120C', '191A', '856E', '997A', '788D', '203C', '110C', '653B', '484C', '956', '362A', '650C', '605D', '503A', '780', '652B', '954', '801', '403A', '697C', '868B', '842E', '793', '861A', '810B', '828A', '221C', '112B', '675D', '230E', '276D', '510A', '407A', '893B', '461B', '961', '892', '355A', '227D', '651B', '486B', '496C', '321C', '866', '84A', '680C', '337B', '216D', '224D', '81A', '339C', '195C', '192B', '324B', '669C', '959', '868C', '199D', '863A', '460B', '230J', '906', '524C', '948', '290B', '94D', '364B', '905', '640A', '491B', '856B', '94E', '689F', '582', '277A', '699A', '995B', '477C', '886', '398', '253A', '356A', '341A', '367A', '491F', '186D', '680B', '418C', '225B', '498', '184C', '494', '893D', '800', '82A', '517E', '990B', '99B', '996A', '785C', '201E', '949', '260D', '495B', '354B', '99A', '717B', '880A', '492B', '438C', '894B', '856D', '7A', '14B', '230D', '857A', '290D', '174C', '123E', '320B', '105D', '42A', '460D', '230G', '863B', '380', '871A', '795A', '99C', '635C', '858B', '133A', '496F', '860A', '226F', '858A', '860B']),
        gr.Dropdown(label="Street Name", choices=['MONTREAL LINK', 'PUNGGOL FIELD WALK', 'BT BATOK WEST AVE 8', 'LOR 1A TOA PAYOH', 'MEI LING ST', 'TAMPINES ST 83', 'YISHUN AVE 11', 'CLEMENTI AVE 1', 'WOODLANDS DR 52', 'HOUGANG AVE 7', 'BOON LAY DR', 'PASIR RIS ST 51', 'JELAPANG RD', 'WEST COAST DR', 'HOUGANG AVE 8', 'BT BATOK WEST AVE 6', 'TAMPINES ST 84', 'PASIR RIS ST 53', 'BT MERAH VIEW', 'BEDOK STH RD', 'CHOA CHU KANG CTRL', 'WOODLANDS CIRCLE', 'HOUGANG ST 51', 'PUNGGOL DR', 'WOODLANDS DR 50', 'BEDOK NTH AVE 2', 'YISHUN AVE 9', 'BEDOK RESERVOIR RD', 'JURONG WEST ST 61', 'PUNGGOL PL', 'PUNGGOL CTRL', 'PASIR RIS ST 71', 'JLN BT HO SWEE', 'COMPASSVALE LANE', 'KEAT HONG LINK', 'HOUGANG AVE 3', 'TAMPINES ST 44', 'GHIM MOH RD', 'TELOK BLANGAH HTS', 'TAMPINES ST 82', 'WOODLANDS DR 14', 'BOON LAY PL', 'YISHUN ST 22', 'CHOA CHU KANG AVE 5', 'BEDOK NTH RD', 'WOODLANDS DR 73', 'WHAMPOA WEST', 'BT BATOK ST 34', 'ANG MO KIO AVE 3', 'WHAMPOA RD', 'WOODLANDS ST 32', 'PUNGGOL WAY', 'YISHUN ST 31', 'PASIR RIS ST 12', 'YISHUN AVE 7', 'ANCHORVALE LINK', 'COMPASSVALE ST', 'BOON KENG RD', 'CHOA CHU KANG NTH 6', 'BT BATOK EAST AVE 5', 'BT BATOK ST 52', 'TAMPINES AVE 9', 'BENDEMEER RD', 'CLEMENTI AVE 4', 'KALLANG BAHRU', 'YISHUN AVE 6', 'RIVERVALE CRES', 'HOUGANG AVE 4', 'JURONG WEST ST 64', 'UPP BOON KENG RD', 'ANCHORVALE RD', 'DORSET RD', 'CHOA CHU KANG AVE 7', 'YISHUN ST 71', 'SERANGOON NTH AVE 4', 'BEDOK NTH ST 1', 'PASIR RIS ST 11', 'SERANGOON AVE 4', 'YISHUN ST 61', 'HOLLAND DR', 'STIRLING RD', 'JURONG WEST ST 65', 'HOUGANG AVE 5', 'TOA PAYOH CTRL', 'ADMIRALTY DR', 'JURONG WEST ST 41', 'TAMPINES ST 22', 'JURONG WEST ST 74', 'REDHILL RD', 'LOR 2 TOA PAYOH', 'COMPASSVALE WALK', 'WOODLANDS ST 82', 'WOODLANDS ST 81', 'SENGKANG EAST AVE', 'ANG MO KIO ST 52', 'PASIR RIS ST 21', 'YISHUN ST 72', 'BUFFALO RD', 'BT BATOK ST 24', 'BT BATOK WEST AVE 5', 'JURONG EAST ST 21', 'YISHUN RING RD', 'BT BATOK ST 33', 'TAMPINES ST 71', 'TELOK BLANGAH CRES', 'TEBAN GDNS RD', 'YISHUN AVE 4', 'BISHAN ST 12', 'SEMBAWANG CL', 'EDGEDALE PLAINS', 'YISHUN AVE 5', 'LOR 1 TOA PAYOH', 'UPP SERANGOON VIEW', 'BEDOK NTH ST 3', 'LOR 7 TOA PAYOH', 'JLN TECK WHYE', 'UPP SERANGOON CRES', 'WOODLANDS AVE 1', 'TAMPINES ST 24', 'LOMPANG RD', 'CHOA CHU KANG CRES', 'HOUGANG AVE 10', 'BT BATOK EAST AVE 3', 'TAMPINES ST 34', 'HOUGANG ST 61', 'TAMPINES AVE 5', 'TG PAGAR PLAZA', 'EDGEFIELD PLAINS', 'SEGAR RD', 'TAMPINES CTRL 8', 'BT BATOK EAST AVE 6', 'BEDOK STH AVE 1', 'HOUGANG ST 22', 'WOODLANDS DR 16', 'EUNOS RD 5', 'SENJA RD', 'BT PANJANG RING RD', 'BEDOK RESERVOIR VIEW', 'TAMPINES ST 81', 'JURONG WEST ST 52', 'CHOA CHU KANG AVE 4', 'UPP SERANGOON RD', 'ANG MO KIO ST 32', 'CANBERRA RD', 'LOR AH SOO', 'CLEMENTI AVE 3', 'DOVER CRES', 'JLN TENTERAM', 'BEDOK NTH AVE 3', 'KEAT HONG CL', 'MOH GUAN TER', 'JURONG WEST ST 24', 'FERNVALE LINK', 'PASIR RIS DR 3', 'MARSILING RISE', 'BOON TIONG RD', 'YISHUN ST 81', 'BOON LAY AVE', 'JURONG EAST ST 24', 'BEDOK STH AVE 3', 'YISHUN AVE 3', 'BUANGKOK CRES', 'ELIAS RD', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 4', 'ANG MO KIO AVE 8', 'PASIR RIS ST 52', 'CHAI CHEE ST', 'YISHUN ST 11', 'PUNGGOL RD', 'JURONG WEST ST 81', 'CHAI CHEE RD', 'KLANG LANE', 'SERANGOON NTH AVE 1', 'PUNGGOL FIELD', 'JURONG WEST ST 91', 'HENDERSON RD', 'TAMPINES ST 43', 'WOODLANDS DR 70', 'ANCHORVALE CRES', 'COMPASSVALE LINK', 'DAWSON RD', 'BISHAN ST 24', 'RIVERVALE DR', 'SEMBAWANG CRES', 'PIPIT RD', 'WOODLANDS DR 62', 'PASIR RIS DR 1', 'SENGKANG WEST WAY', 'TELOK BLANGAH DR', 'SIMS DR', 'YISHUN ST 51', 'FAJAR RD', 'SERANGOON AVE 2', 'WOODLANDS DR 44', 'STRATHMORE AVE', 'CHAI CHEE AVE', 'PETIR RD', 'FERNVALE LANE', 'SIMEI RD', 'COMPASSVALE BOW', 'TAO CHING RD', 'JURONG WEST ST 25', 'FERNVALE RD', "ST. GEORGE'S LANE", 'TAMPINES ST 91', 'SIMEI ST 1', 'TAMPINES ST 45', 'JELEBU RD', 'BT BATOK WEST AVE 9', 'PASIR RIS DR 10', 'CANTONMENT RD', 'WOODLANDS DR 40', 'BT BATOK ST 25', 'CANBERRA ST', 'PASIR RIS DR 6', 'BT BATOK ST 22', 'WATERLOO ST', 'WOODLANDS RING RD', 'ANG MO KIO AVE 1', 'KIM KEAT AVE', 'YISHUN ST 41', 'WOODLANDS CRES', 'WOODLANDS DR 60', 'CLEMENTI WEST ST 2', 'BT BATOK ST 21', 'BEDOK STH AVE 2', 'BISHAN ST 13', 'CHOA CHU KANG ST 62', 'LENGKOK BAHRU', 'TAMPINES ST 32', 'ALJUNIED CRES', 'POTONG PASIR AVE 1', 'MARINE TER', 'TAMPINES CTRL 7', 'WHAMPOA DR', 'TAMPINES ST 21', 'MARSILING RD', 'TOH YI DR', 'CIRCUIT RD', 'YUAN CHING RD', 'YUNG PING RD', 'HOLLAND CL', 'UBI AVE 1', 'JLN BT MERAH', 'YISHUN CTRL', 'BUANGKOK GREEN', 'CHOA CHU KANG AVE 1', 'WOODLANDS ST 41', 'CLEMENTI AVE 2', 'CLEMENTI WEST ST 1', 'TELOK BLANGAH ST 31', 'HOUGANG AVE 1', 'CASSIA CRES', 'CHOA CHU KANG ST 64', 'SUMANG WALK', 'TOA PAYOH NTH', 'SERANGOON AVE 3', 'KIM TIAN RD', 'PASIR RIS DR 4', 'ANCHORVALE DR', 'NEW UPP CHANGI RD', 'DEPOT RD', 'PAYA LEBAR WAY', 'WOODLANDS DR 75', 'JLN BAHAGIA', 'BT BATOK CTRL', 'TAMPINES ST 33', 'HOUGANG AVE 6', 'WOODLANDS AVE 4', 'LOR 4 TOA PAYOH', 'TELOK BLANGAH RISE', 'SIMEI ST 4', 'YISHUN AVE 1', 'CANBERRA CRES', 'BISHAN ST 22', 'PUNGGOL WALK', 'PANDAN GDNS', 'TAMPINES ST 12', 'WOODLANDS RISE', 'JURONG WEST ST 93', 'ANG MO KIO AVE 5', 'GHIM MOH LINK', 'TAMPINES ST 61', 'JLN TENAGA', 'YISHUN AVE 2', 'HO CHING RD', 'WOODLANDS AVE 5', 'BT PURMEI RD', 'LOR 5 TOA PAYOH', 'COMPASSVALE CRES', "C'WEALTH AVE WEST", 'SENGKANG EAST WAY', 'WOODLANDS ST 11', 'JURONG WEST CTRL 1', 'BEDOK NTH AVE 4', 'JURONG WEST ST 42', 'ANCHORVALE ST', 'SENGKANG CTRL', 'WEST COAST RD', 'SUMANG LANE', 'WOODLANDS AVE 6', 'SIMEI LANE', 'CAMBRIDGE RD', "C'WEALTH CL", 'LOR LIMAU', 'CHOA CHU KANG AVE 3', 'TECK WHYE AVE', 'HENDERSON CRES', 'MARSILING LANE', 'CHOA CHU KANG AVE 2', 'JURONG EAST ST 13', 'CHOA CHU KANG ST 52', 'MOULMEIN RD', 'ANG MO KIO ST 51', 'BT BATOK ST 31', 'CORPORATION DR', 'YUNG SHENG RD', 'JURONG EAST AVE 1', 'SIN MING AVE', 'ANG MO KIO ST 44', 'CLEMENTI AVE 6', 'TAMPINES AVE 1', 'PASIR RIS ST 13', 'COMPASSVALE DR', 'JURONG EAST ST 31', 'SENGKANG WEST AVE', 'BISHAN ST 23', 'PUNGGOL EAST', 'BUANGKOK LINK', 'LOR 8 TOA PAYOH', 'LENGKONG TIGA', 'SEMBAWANG DR', 'HOUGANG ST 91', 'TAMPINES ST 72', 'RACE COURSE RD', 'SERANGOON CTRL', 'JURONG WEST ST 71', 'TAMPINES ST 42', 'TOH GUAN RD', "C'WEALTH DR", 'PENDING RD', 'HOUGANG ST 11', 'MCNAIR RD', 'LOR 3 TOA PAYOH', 'RIVERVALE WALK', 'SERANGOON NTH AVE 3', 'MARSILING DR', "C'WEALTH CRES", 'MARINE CRES', 'ANG MO KIO ST 11', 'MARINE DR', 'MACPHERSON LANE', 'CLEMENTI AVE 5', 'EUNOS CRES', 'CHOA CHU KANG ST 54', 'WELLINGTON CIRCLE', 'KIM KEAT LINK', 'ANG MO KIO AVE 2', 'EVERTON PK', 'TAMPINES AVE 7', 'BEDOK NTH ST 2', 'HOUGANG ST 92', 'SIN MING RD', 'SERANGOON AVE 1', 'YISHUN ST 21', 'TECK WHYE LANE', 'HOUGANG AVE 9', 'BT BATOK WEST AVE 4', 'CASHEW RD', 'HAVELOCK RD', 'FARRER PK RD', 'DOVER RD', 'JLN MEMBINA', 'TAMPINES AVE 4', 'JURONG WEST AVE 3', 'BT BATOK ST 32', 'CHOA CHU KANG LOOP', 'CANBERRA WALK', 'TAMPINES ST 23', 'JURONG EAST ST 32', 'CHOA CHU KANG DR', 'JURONG WEST ST 75', 'BEACH RD', 'YISHUN ST 20', 'ANG MO KIO ST 31', 'COMPASSVALE RD', 'CHANGI VILLAGE RD', 'JURONG WEST ST 92', 'JURONG WEST AVE 1', 'BAIN ST', 'TAMPINES AVE 8', 'BT BATOK ST 11', 'RIVERVALE ST', 'TAMPINES ST 41', 'WOODLANDS ST 83', 'CHOA CHU KANG ST 51', 'CHOA CHU KANG ST 53', 'BEDOK NTH AVE 1', 'SAUJANA RD', "ST. GEORGE'S RD", 'SHUNFU RD', 'SERANGOON NTH AVE 2', 'GEYLANG BAHRU', "QUEEN'S RD", 'WOODLANDS DR 72', 'GANGSA RD', 'JLN RAJAH', 'CHOA CHU KANG NTH 5', 'JLN BERSEH', 'GEYLANG EAST CTRL', 'WHAMPOA STH', 'TAMPINES ST 11', 'OLD AIRPORT RD', 'GEYLANG SERAI', 'TOA PAYOH EAST', 'WOODLANDS ST 13', 'BEO CRES', 'HOUGANG CTRL', 'GEYLANG EAST AVE 1', 'KENT RD', 'CRAWFORD LANE', 'TAH CHING RD', 'SUMANG LINK', 'MARSILING CRES', 'HOLLAND AVE', 'JLN RUMAH TINGGI', 'BALAM RD', 'MONTREAL DR', 'VEERASAMY RD', 'JURONG WEST AVE 5', 'ANG MO KIO AVE 6', 'AH HOOD RD', 'CANBERRA LINK', 'JLN KUKOH', 'HOUGANG ST 52', 'WOODLANDS ST 31', 'ADMIRALTY LINK', 'WOODLANDS DR 71', 'YUNG AN RD', 'JURONG WEST ST 73', 'UPP ALJUNIED LANE', 'KANG CHING RD', 'POTONG PASIR AVE 3', 'REDHILL CL', 'DAKOTA CRES', 'BT MERAH CTRL', 'SEMBAWANG VISTA', 'JOO SENG RD', 'LOR LEW LIAN', 'BT BATOK EAST AVE 4', 'CANTONMENT CL', 'SERANGOON CTRL DR', 'INDUS RD', 'TANGLIN HALT RD', 'BEDOK NTH ST 4', 'BEDOK RESERVOIR CRES', 'HOUGANG ST 31', 'HAIG RD', 'TIONG BAHRU RD', 'CLEMENTI ST 14', 'DELTA AVE', 'WOODLANDS AVE 3', 'BANGKIT RD', 'PINE CL', 'YUNG KUANG RD', 'WOODLANDS DR 42', 'JELLICOE RD', 'SMITH ST', "JLN MA'MOR", 'ANCHORVALE LANE', 'SPOTTISWOODE PK RD', 'CLARENCE LANE', 'FERNVALE ST', 'YUNG LOH RD', 'FARRER RD', 'BISHAN ST 11', 'POTONG PASIR AVE 2', 'JLN TIGA', 'TAMPINES CTRL 1', 'ALJUNIED RD', 'JURONG WEST ST 72', 'ROWELL RD', 'HOUGANG AVE 2', 'CHAI CHEE DR', 'EMPRESS RD', 'KELANTAN RD', 'SELEGIE RD', 'HOUGANG ST 21', "QUEEN'S CL", 'KIM TIAN PL', 'SIMEI ST 2', 'LOR 6 TOA PAYOH', 'JLN BATU', 'SENJA LINK', 'HOY FATT RD', 'JLN KAYU', 'BEDOK CTRL', 'SIMEI ST 5', 'BT BATOK WEST AVE 2', 'CHOA CHU KANG NTH 7', 'OWEN RD', 'ANG MO KIO AVE 9', 'LIM LIAK ST', 'YISHUN ST 43', 'TESSENSOHN RD', 'JLN KLINIK', 'REDHILL LANE', 'JLN DAMAI', "KING GEORGE'S AVE", 'TOWNER RD', 'BRIGHT HILL DR', 'GLOUCESTER RD', 'DOVER CL EAST', 'JURONG WEST CTRL 3', 'CLEMENTI ST 13', 'SENGKANG EAST RD', 'TAMPINES ST 86', 'WOODLANDS DR 53', 'YUNG HO RD', 'SIMS PL', 'YISHUN CTRL 1', 'BT BATOK WEST AVE 7', 'JLN DUA', 'NTH BRIDGE RD', 'BT BATOK ST 51', 'CLEMENTI ST 12', "C'WEALTH AVE", 'JURONG WEST ST 62', 'TECK WHYE CRES', 'JLN DUSUN', 'UPP CROSS ST', 'WOODLANDS AVE 9', 'PASIR RIS ST 72', 'HOUGANG ST 32', 'TELOK BLANGAH WAY', 'CLEMENTI ST 11', 'JOO CHIAT RD', 'BT MERAH LANE 1', 'SIMS AVE', 'LOR 3 GEYLANG', 'SAGO LANE', 'KIM CHENG ST', 'PASIR RIS ST 41', 'TAMAN HO SWEE', 'CHIN SWEE RD', 'ANG MO KIO ST 21', 'QUEEN ST', 'LOWER DELTA RD', 'ANG MO KIO ST 61', 'KG ARANG RD', 'KIM PONG RD', 'JURONG WEST ST 51', 'SILAT AVE', 'SENG POH RD', 'ZION RD', 'KG KAYU RD', 'QUEENSWAY', 'CHANDER RD', 'KRETA AYER RD', 'FRENCH RD', 'NEW MKT RD', 'GEYLANG EAST AVE 2', 'SEMBAWANG WAY', 'MARINE PARADE CTRL', 'ALJUNIED AVE 2']),   
        gr.Dropdown(label="Storey Range", choices=['16 TO 18', '10 TO 12', '04 TO 06', '13 TO 15', '22 TO 24',
       '07 TO 09', '37 TO 39', '01 TO 03', '28 TO 30', '25 TO 27',
       '19 TO 21', '34 TO 36', '31 TO 33', '40 TO 42', '43 TO 45',
       '46 TO 48', '49 TO 51']), 
        gr.Number(label="Floor Area (sqm)"),
        gr.Dropdown(label="Flat Model", choices=['Model A', 'Improved', 'New Generation', 'DBSS', 'Simplified',
       'Apartment', 'Standard', 'Premium Apartment', 'Maisonette',
       'Model A-Maisonette', 'Premium Apartment Loft',
       'Premium Maisonette', 'Type S1', 'Model A2', 'Type S2', '2-room',
       'Terrace', 'Adjoined flat', 'Improved-Maisonette',
       'Multi Generation', '3Gen']),  
        gr.Dropdown(choices=years, label="Lease Commence Date"),
        gr.Number(label="Remaining Lease Years", precision=0),
        gr.Dropdown(label="Remaining Lease Months", choices=months),
    ],
    outputs=gr.Text(label="Predicted flat price"),
)

if __name__ == "__main__":
    demo.launch(server_port=5155, server_name="0.0.0.0")