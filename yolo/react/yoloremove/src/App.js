import React, { Component } from 'react';
import './App.css';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import { render } from "react-dom";
import { BrowserRouter } from "react-router-dom";
import { Switch, Route } from "react-router-dom";
import { browserHistory } from "react-router";
import { Redirect } from "react-router-dom";
import { Link } from "react-router-dom";
import TextField from "@material-ui/core/TextField";
import { withRouter } from 'react-router-dom';
import Button from "@material-ui/core/Button";
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import Paper from '@material-ui/core/Paper';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import Stepper from '@material-ui/core/Stepper';
import Step from '@material-ui/core/Step';
import StepLabel from '@material-ui/core/StepLabel';
import StepContent from '@material-ui/core/StepContent';
import FolderIcon from "@material-ui/icons/Folder";
import LinearProgress from "@material-ui/core/LinearProgress";
import CircularProgress from '@material-ui/core/CircularProgress';
import Fab from '@material-ui/core/Fab';
import Input from '@material-ui/core/Input';
import OutlinedInput from '@material-ui/core/OutlinedInput';
import FilledInput from '@material-ui/core/FilledInput';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import Slider from '@material-ui/lab/Slider';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import firebase from './Firebase';
import sample01 from './IMG_7505.jpg';
import sample_crop from './public_IMG_7505_0_dog.png';
import axios from 'axios';


const styles = {
  root: {
    flexGrow: 1,
  },
  
  slider: {
    padding: '22px 0px',
	 
  },
  gridList: {
    width: 300,
  },
  titleBar: {
    background:
      'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, ' +
      'rgba(0,0,0,0.3) 70%, rgba(0,0,0,0) 100%)',
  },
};

class App extends Component {
  state = {
    open: false,
	objlist: [],
	filename: "",
    activeStep: 0,
    uploading: false,
    loaded: false,
    imgdata: null,
	dynvalue: 25,
	errormsg: "",
	selectname: "",
	cropname: "",
	cropimg: null,
	bgname: "",
	bgimg: null,
  };
  uuidv4() {
	  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
		var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
		return v.toString(16);
	  });
  }
    getLast(str) {
		var arr = str.split('_')
		return arr[arr.length - 1]
	}

   handleChange = name => event => {
	console.log("select ", event.target.value);
	var obj = event.target.value
	var hosturl = "https://storage.cloud.google.com/iwasnothing03.appspot.com/" 
	//var imgsrc = "https://storage.cloud.google.com/iwasnothing03.appspot.com/" + event.target.value; 
	var imgsrc = hosturl + obj
	var arr = obj.split('.')
	var crop = hosturl + arr[0] + '_crop.' + arr[1]
    this.setState({
	  selectname: this.getLast(event.target.value),
	  imgdata: imgsrc,
	  cropname: crop,
	  cropimg: crop,
	  bgname: imgsrc,
	  bgimg:  imgsrc,
	  activeStep: 2,
    });

  };
  goNext = () => {
	  var x = this.state.activeStep;
	  this.setState({activeStep: x + 1})
  };
  handleChange2 = (event,value) => {
	console.log("slide ", value);
	this.setState({dynvalue: value});
  };
  doProcess = (method,val1,val2,val3,val4) => {
	 var self = this;
	 var filename = this.state.filename
	//axios.defaults.timeout =  3600;
	axios.post("https://iwasnothing03.appspot.com/image", {'file': filename, 'method': method, 'parm1': val1, 'parm2': val2, 'parm3': val3, 'parm4': val4})
      .then(res => {
        const list = res.data;
		console.log("generated img ", list);
		var items = list['items']
		console.log("items: ",items)

        self.setState({uploading: false, loaded: true, activeStep: 1,  objlist: items })
      })
	  .catch(error => {
			console.log(error);
			self.setState({uploading: false, loaded: true, errormsg: "image processing error"})
		});
  }
  handleHist = () => {
	console.log("grab ")
	var x = this.state.dynvalue/3
	var dynval1 = 1
	var dynval2 = 5
	var dynval3 = 5
	var dynval4 = x
	this.setState({uploading: true,loaded: false, activeStep: 0,imgdata: "", selectname: "", cropname: "", cropimg: "", bgname: "", bgimg: ""});
	this.doProcess('histcut',dynval1,dynval2,dynval3,dynval4)
  };
    handleReset = () => {
	console.log("rerun ")
	this.setState({uploading: false,loaded: false, activeStep: 0,imgdata: "", selectname: "", filename: "", cropname: "", cropimg: "", bgname: "", bgimg: ""});
	
  };
  handleGrab = () => {
	console.log("grab ")
	var x = this.state.dynvalue/30
	var dynval1 = 1
	var dynval2 = 5
	var dynval3 = 5
	var dynval4 = x
	this.setState({uploading: true,loaded: false, activeStep: 0,imgdata: "", selectname: "", cropname: "", cropimg: "", bgname: "", bgimg: ""});
	this.doProcess('grabcut',dynval1,dynval2,dynval3,dynval4)
  };

  handleRerun = () => {
	console.log("rerun ")
	var x = this.state.dynvalue/10
	var dynval1 = x
	var dynval2 = x
	var dynval3 = x
	var dynval4 = x
	this.setState({uploading: true,loaded: false, activeStep: 0,imgdata: "", selectname: "", cropname: "", cropimg: "", bgname: "", bgimg: ""});
	this.doProcess('histcut',1,10,5,dynval4)
  };
   
  downloadFile(namefield,field,filename) {
    var self = this;
	var objname = filename.split('/')[1]
    console.log("download file",objname)
    var storageRef = firebase.storage().ref('public');
    var ImagesRef = storageRef.child(objname);
    ImagesRef.getDownloadURL().then(function(url) { 
      console.log('download a blob or file!');
	  self.setState({[field]: url, [namefield]: url})
      
    })
	.catch(function(error) {
		console.log(error);
        self.setState({errormsg: "download error"})
        });
	
  }
  uploadFile(event) {
    var self = this;
	var x = 5;
	self.setState({uploading: true,loaded: false});
    const file = event.target.files[0];
    console.log("upload file",file)
    this.setState({uploading: true,loaded: false, filename: file.name});
    var storageRef = firebase.storage().ref('public');
    var ImagesRef = storageRef.child(file.name);
    ImagesRef.put(file).then(function(snapshot) {
      console.log('Uploaded a blob or file!');
	  self.doProcess('histcut',1,10,5,1)
      
    })
	.catch(function(error) {
		console.log(error);
        self.setState({uploading: false, loaded: true, errormsg: "upload error"})
        });
	
  }
  render() {
	const { classes } = this.props;
	const imgname = this.state.imgdata;
	const  activeStep  = this.state.activeStep;
    return (
    <div className={classes.root}>
      <AppBar position="static" color="default">
        <Toolbar>
          <Typography variant="h6" color="inherit">
            Remove Photo Background
          </Typography>
        </Toolbar>
      </AppBar>
	  		<div style={{width: 300, marginLeft: 50, marginTop: 50}}>
		<GridList cellHeight={200}  cols={2}>
		<GridListTile rows={1.7}  cols={2}>
	  <div><ol>
	  <li type="1">The system will automatically detect every object in the photo, and change the background of each object to transparent.  </li>
	  <li type="1">The aim is to facilitate the making of WhatsApp stickers. </li>
	  <li type="1">We use color histogram to differentiate the background from the subject.  </li>
	  <li type="1">If the color contrast between the subject and the background is clear enough, it will generate a good output, just like the sample below.</li>
	  <li type="1">If the color of the subject is close to the background, it will fail.</li>
	 </ol> </div>
	  </GridListTile>
	  <GridListTile cols={1}>
	  <img src={sample01} width="140" height="200"  />
	  </GridListTile>
	  <GridListTile cols={1}>
	  <img src={sample_crop} width="140" height="200"/>
	   </GridListTile>
	  </GridList>
	  </div>
	
	        <Stepper activeStep={activeStep} orientation="vertical">
            <Step key="Upload Photo">
            <StepLabel>Upload Photo</StepLabel>
            <StepContent>
			  <div>
                <input
                  id="myInput"
                  type="file"
                  ref={ref => (this.upload = ref)}
                  style={{ display: "none" }}
                  onChange={this.uploadFile.bind(this)}
                />
              </div>
              <div style={{ width: 100, margin: 10 }}>
                <Fab size="medium"
                  color="primary"
                  aria-label="add"
                  style={{ margin: 10 }}
                  onClick={e => this.upload.click()}
                >
                  <FolderIcon />
                </Fab>
              </div>
              {this.state.uploading ? ( <div>
                <CircularProgress color="primary" />
				<Typography variant="caption" color="inherit">
					Please wait for about 1-2 min
				  </Typography></div>
              ) : (
                <div />
              )}
			 {this.state.loaded ? ( <div>
				<Typography variant="caption" color="inherit">
				{this.state.errormsg}
				  </Typography></div>
              ) : (
                <div />
              )}
				
              </StepContent>
              </Step>
			<Step key="Select Object">
            <StepLabel>Select Object  in the list below</StepLabel>
            <StepContent>
		<FormControl style={{width: 200}}>
		<InputLabel >Please Select the Object</InputLabel>
		<Typography variant="caption" color="inherit"> Detected {this.state.objlist.length} Objects</Typography>
          <Select
            value={this.state.selectname}
			onChange={this.handleChange('selectname')}
          >
		  
		  {this.state.objlist.map(obj => {
         return (
            <MenuItem value={obj.name}>{this.getLast(obj.name)}</MenuItem>
		  ) } ) }
          </Select>
        </FormControl>
		     </StepContent>
              </Step>
		<Step key="View Photo">
            <StepLabel>View the result below</StepLabel>
            <StepContent>
		<Typography variant="h6" color="inherit">
			If you cannot view the image, please click download link.
		</Typography>
		<br/>

            <Typography variant="h6" color="inherit">Cropped Image (<a href={this.state.cropname} target="_blank">Download image</a>)</Typography>
			 
			 <img src={this.state.cropimg} />
		<br/>
		<br/>
            <Typography variant="h6" color="inherit">Removed Background (<a href={this.state.bgname} target="_blank" >Download image</a>)</Typography>
			 
		
				<img src={this.state.bgimg} />
			<br/>	
			<Button
				variant="outlined"
				color="primary"
				onClick={this.handleReset}
			  >
			  Start Over
			  </Button>
			  <Button
				variant="outlined"
				color="primary"
				onClick={this.goNext}
			  >
			  Next
			  </Button>
	 		     </StepContent>
              </Step>
		<Step key="Retune">
            <StepLabel>Fine Tune</StepLabel>
            <StepContent>


		<Typography variant="h6" color="inherit">
			Adjust the slider and run fine tune again.
		</Typography>
		<Button
				variant="outlined"
				color="primary"
				onClick={this.handleRerun}
			  >
			  fine tune
			  </Button>

		<div style={{width: 300}}>
		<Slider
          classes={{ container: classes.slider }}
		  min={0}
          max={10}
          value={this.state.dynvalue}
          aria-labelledby="label"
          onChange={this.handleChange2}
        />
		</div>
		<div style={{width: 300}}>
		<GridList cellHeight={40} className={classes.gridList} cols={3}>
		<GridListTile>
		<Typography variant="caption" color="inherit">more background</Typography>
		</GridListTile>
		<GridListTile></GridListTile>
		<GridListTile>
		<Typography variant="caption" color="inherit">less background</Typography>
		</GridListTile>
		</GridList>
		</div>
		<br/>
			<Button
				variant="outlined"
				color="primary"
				onClick={this.handleReset}
			  >
			  Start Over
			  </Button>
			  <Button
				variant="outlined"
				color="primary"
				onClick={this.goNext}
			  >
			  Finish
			  </Button>
			</StepContent>
              </Step>
			<Step key="finish">
            <StepLabel>Finish</StepLabel>
            <StepContent>
<img src={this.state.bgimg} />
<a href={this.state.bgname} target="_blank" >Download image</a>
	  <div><ol>
	  <li type="1">Save the above image</li>
	  <li type="1">Download the following App to import the sticker to your WhatsApp</li>
		</ol></div>
		<br/><a href="https://itunes.apple.com/us/app/wstick/id1442273161?mt=8">IOS App</a>
		<br/><a href="https://play.google.com/store/apps/details?id=com.dstukalov.walocalstoragestickers&hl=zh_HK">Android App</a>
				<br/>	<br/><Button
				variant="outlined"
				color="primary"
				onClick={this.handleReset}
			  >
			  Start Over
			  </Button>
			  </StepContent>
              </Step>
              </Stepper>
	  
    </div>
    );
  }
}

export default  withStyles(styles)(App);
