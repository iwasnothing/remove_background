import * as firebase from 'firebase';
import firestore from 'firebase/firestore'

const settings = {timestampsInSnapshots: true};

const config = {
    apiKey: "AIzaSyC7Lc9kmrSMPy2Y-TWNr7kw2-Y1aR5iVxU",
    authDomain: "iwasnothing03.firebaseapp.com",
    databaseURL: "https://iwasnothing03.firebaseio.com",
    projectId: "iwasnothing03",
    storageBucket: "iwasnothing03.appspot.com",
    messagingSenderId: "877799303116"
};
firebase.initializeApp(config);

firebase.firestore().settings(settings);

export default firebase;
