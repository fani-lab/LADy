import Head from "next/head";
import { Inter } from "@next/font/google";
import {
  Box,
  Center,
  Container,
  Divider,
  HStack,
  Heading,
  ListItem,
  Text,
  UnorderedList,
  VStack,
} from "@chakra-ui/layout";
import { Textarea } from "@chakra-ui/textarea";
import { InfoOutlineIcon, RepeatIcon, SmallCloseIcon } from "@chakra-ui/icons";

import Footer from "./Components/Footer";
import { useState } from "react";
import { Button } from "@chakra-ui/button";
import Example from "./Components/Chart";
import {
  Select,
  FormControl,
  FormLabel,
  FormErrorMessage,
  FormHelperText,
} from "@chakra-ui/react";
const inter = Inter({ subsets: ["latin"] });
//use state to store textarea value
export default function Home() {
  const [formval, setformval] = useState("");
  function handleSubmit(e) {
    e.preventDefault();
    console.log(formval);
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: formval,
        model: "ree",
      }),
    };

    fetch("http://localhost:5000/api", requestOptions).then((response) =>
      response.json().then((data) => console.log(data))
    );
  }
  let handleInputChange = (e) => {
    let inputValue = e.target.value;
    setformval(inputValue);
  };
  const isError = formval === "";
  return (
    <>
      <Head>
        <title>Latent Aspect Detection</title>
        <meta name="description" content="Generated by create next app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Container minWidth={"container.lg"} p="5">
        <Heading mb="10">Latent Aspect Detection</Heading>
        <FormControl isInvalid={isError}>
          <HStack mb="5" spacing={4}>
            <Button
              leftIcon={<InfoOutlineIcon />}
              colorScheme="teal"
              variant="outline"
            >
              Info
            </Button>{" "}
            <Button
              leftIcon={<RepeatIcon />}
              colorScheme="teal"
              variant="outline"
            >
              Random review
            </Button>
            <FormLabel>Model: </FormLabel>
            <Select maxWidth={"200px"} borderColor={"teal"}>
              <option value="option1">Option 1</option>
              <option value="option2">Option 2</option>
              <option value="option3">Option 3</option>
            </Select>
          </HStack>
          <Textarea
            mb="5"
            placeholder="Here is a sample placeholder"
            onChange={handleInputChange}
            value={formval}
          />
          <Button colorScheme="teal" onClick={handleSubmit}>
            Submit
          </Button>
        </FormControl>
        <Example />

        <Footer />
      </Container>
    </>
  );
}