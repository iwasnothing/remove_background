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
import firebase from './Firebase';
import axios from 'axios';


const styles = {
  root: {
    flexGrow: 1,
  },
  
  slider: {
    padding: '22px 0px',
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
	selectname: ""
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
	var imgsrc = "https://storage.cloud.google.com/iwasnothing03.appspot.com/" + event.target.value; 
    this.setState({
	  selectname: this.getLast(event.target.value),
	  imgdata: imgsrc
    });
  };
  handleChange2 = (event,value) => {
	console.log("slide ", value);
	this.setState({dynvalue: value});
  };
  doProcess = (val1,val2,val3,val4) => {
	 var self = this;
	 var filename = this.state.filename
	axios.post("https://iwasnothing03.appspot.com/image", {'file': filename, 'parm1': val1, 'parm2': val2, 'parm3': val3, 'parm4': val4})
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
  handleRerun = () => {
	console.log("rerun ")
	var dynval1 = this.state.dynvalue
	var dynval2 = this.state.dynvalue
	var dynval3 = this.state.dynvalue
	var dynval4 = this.state.dynvalue
	this.setState({uploading: true,loaded: false, activeStep: 0,imgdata: "", filename: ""});
	this.doProcess(dynval1,dynval2,dynval3,dynval4)
  };
    handleReset = () => {
	console.log("rerun ")
	this.setState({uploading: false,loaded: false, activeStep: 0,imgdata: "", filename: ""});
	
  };
  uploadFile(event) {
    var self = this;
	self.setState({uploading: true,loaded: false});
    const file = event.target.files[0];
    console.log("upload file",file)
    this.setState({uploading: true,loaded: false, filename: file.name});
    var storageRef = firebase.storage().ref('public');
    var ImagesRef = storageRef.child(file.name);
    ImagesRef.put(file).then(function(snapshot) {
      console.log('Uploaded a blob or file!');
	  self.doProcess(25,22,25,25)
      
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
            Photos
          </Typography>
        </Toolbar>
      </AppBar>
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
				<Typography variant="h6" color="inherit">
					Please wait for about 20 sec
				  </Typography></div>
              ) : (
                <div />
              )}
			 {this.state.loaded ? ( <div>
				<Typography variant="h6" color="inherit">
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
		<InputLabel >Please Select the Object </InputLabel>
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
		<br/>
		<a href={this.state.imgdata}>Download image</a>
		<br/>
		<img src={this.state.imgdata}/>
		<br/>

		<Typography variant="h6" color="inherit">
			Adjust the slider and run re-tune again
		</Typography>
		<Button
				variant="outlined"
				color="primary"
				onClick={this.handleRerun}
			  >
			  Re-tune
			  </Button>
		<Slider
          classes={{ container: classes.slider }}
          value={this.state.dynvalue}
          aria-labelledby="label"
          onChange={this.handleChange2}
        />

			<Button
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
