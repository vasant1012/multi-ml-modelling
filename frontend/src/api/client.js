import axios from "axios";import { environment } from "../config/environment";
const client=axios.create({baseURL:environment.apiBaseUrl,timeout:120000,headers:{"Content-Type":"application/json"}});
export default client;
